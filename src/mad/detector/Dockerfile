FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WORKER_THREADS=4
ENV DB_HOST=80.93.60.49
ENV DB_NAME=ml_models
ENV DB_USER=mad
ENV DB_PASSWORD=secretPASSW0rd
ENV VICTORIA_METRICS_URL=http://80.93.60.49:30429/api/v1

CMD ["python", "-u", "main.py"]