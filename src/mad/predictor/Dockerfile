FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-warn-script-location -r requirements.txt

CMD ["python", "main.py"]