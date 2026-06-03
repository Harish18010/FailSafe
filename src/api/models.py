from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active       = Column(Boolean, default=True)

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id                = Column(Integer, primary_key=True, index=True)
    batch_id          = Column(String, index=True)
    student_id        = Column(String, index=True)
    risk_probability  = Column(Float)
    is_at_risk        = Column(Boolean)
    intervention_plan = Column(String)
    actioned          = Column(Boolean, default=False)
    actioned_by       = Column(String, nullable=True)
    actioned_at       = Column(DateTime(timezone=True), nullable=True)
    created_at        = Column(DateTime(timezone=True), server_default=func.now())

class InterventionRule(Base):
    __tablename__ = "intervention_rules"
    id        = Column(Integer, primary_key=True, index=True)
    feature   = Column(String, index=True)
    operator  = Column(String)
    threshold = Column(Float)
    plan_text = Column(String)
    is_active = Column(Boolean, default=True)