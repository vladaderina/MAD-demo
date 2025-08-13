FROM python:3.9-slim

WORKDIR /app

COPY config/ /app/config/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

VOLUME /app/config

ENTRYPOINT ["python", "main.py"]