FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей для TensorFlow
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Сначала устанавливаем совместимую версию protobuf
RUN pip install protobuf==3.20.3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]