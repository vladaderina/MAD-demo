FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем структуру директорий
RUN mkdir -p /app/config /app/data/output

# Копируем код
COPY app/ ./collector/
COPY config/ ./config/

CMD ["python", "-m", "collector.main"]