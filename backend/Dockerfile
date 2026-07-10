FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

ENV INPUT_PATH=/input/tasks.json
ENV OUTPUT_PATH=/output/results.json

WORKDIR /app/app
CMD ["python", "main.py"]