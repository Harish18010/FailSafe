import os
from sqlalchemy.orm import Session
from models import InterventionRule
from google import genai
from google.genai import types

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _get_rule_flags(student_data: dict, rules: list) -> list[str]:
    flags = []
    for rule in rules:
        value = student_data.get(rule.feature)
        if value is None:
            continue
        try:
            val_float = float(value)
        except (ValueError, TypeError):
            continue
        if (rule.operator == '>'  and val_float >  rule.threshold) or \
           (rule.operator == '<'  and val_float <  rule.threshold) or \
           (rule.operator == '==' and val_float == rule.threshold):
            flags.append(rule.plan_text)
    return flags


def _build_prompt(student_data: dict, flags: list[str]) -> str:
    # Works with both full student profiles AND the frontend payload
    # (which sends student_id, risk_prob, risk_factors list)
    risk_factors = student_data.get("risk_factors", [])
    risk_prob    = student_data.get("risk_prob", "unknown")
    student_id   = student_data.get("student_id", "unknown")

    # Format SHAP risk factors if present
    if risk_factors:
        factors_text = "\n".join(
            f"- {f['feature']}: SHAP impact {f['impact']:+.4f} ({'increases' if f['impact'] > 0 else 'decreases'} risk)"
            if isinstance(f, dict) else f"- {f}"
            for f in risk_factors
        )
    else:
        factors_text = "- No SHAP data available."

    flags_text = "\n".join(f"- {f}" for f in flags) if flags else "- No rule flags triggered."

    # Include any extra profile fields if available
    profile_keys = [
        "age", "sex", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "traveltime", "studytime", "failures", "schoolsup", "famsup",
        "paid", "activities", "higher", "internet", "famrel", "freetime",
        "goout", "Dalc", "Walc", "health", "absences", "grade_trend", "mid_avg",
    ]
    profile = {k: student_data[k] for k in profile_keys if k in student_data}
    profile_text = str(profile) if profile else "Full profile not available — use SHAP factors below."

    return f"""You are an academic intervention specialist at a secondary school.
Student {student_id} has been flagged as at-risk of failure (risk score: {risk_prob}%).

STUDENT PROFILE:
{profile_text}

TOP SHAP RISK DRIVERS (model explanation):
{factors_text}

RULE-BASED FLAGS:
{flags_text}

Write a concise, actionable, personalised intervention plan.
Structure:
1. Root cause summary (2 sentences)
2. Immediate actions (3 bullet points)
3. 4-week monitoring plan (2 sentences)

Be specific to this student's data. Address the faculty member directly.
Keep total response under 200 words."""


def generate_intervention(student_data: dict, db: Session) -> str:
    rules = db.query(InterventionRule).filter(InterventionRule.is_active == True).all()
    flags = _get_rule_flags(student_data, rules)
    try:
        client = _get_client()
        prompt = _build_prompt(student_data, flags)
        full_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048),
        ):
            if chunk.text:
                full_text += chunk.text
        return full_text.strip() if full_text else _fallback(flags)
    except Exception as e:
        print(f"[intervention] Gemini call failed ({e}), using rule-based fallback.")
        return _fallback(flags)


def stream_intervention(student_data: dict, db: Session):
    rules = db.query(InterventionRule).filter(InterventionRule.is_active == True).all()
    flags = _get_rule_flags(student_data, rules)
    try:
        client = _get_client()
        prompt = _build_prompt(student_data, flags)
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048),
        ):
            if chunk.text:
                # Replace newlines so they don't break SSE framing
                safe = chunk.text.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"[intervention] Stream failed ({e}), sending fallback.")
        yield f"data: {_fallback(flags)}\n\n"
        yield "data: [DONE]\n\n"


def _fallback(flags: list[str]) -> str:
    return " | ".join(flags) if flags else "Standard monitoring. No immediate critical flags."