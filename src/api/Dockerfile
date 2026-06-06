FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/ .
COPY models/ ./models/

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]