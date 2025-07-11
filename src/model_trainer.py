from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import psycopg2
import os
import json

app = FastAPI()

DB_URL = os.environ["DB_URL"]  # postgresql://user:pass@host/dbname

class TimeSeriesPayload(BaseModel):
    name: str
    labels: Dict[str, str]
    values: List[List[float]]  # [timestamp, value]

def build_model(input_shape, config):
    model = Sequential()
    for layer in config["layers"]:
        if layer["type"] == "LSTM":
            model.add(LSTM(units=layer["units"], return_sequences=layer.get("return_sequences", False), input_shape=input_shape))
        elif layer["type"] == "Dense":
            model.add(Dense(units=layer["units"]))
        if config.get("dropout"):
            model.add(Dropout(config["dropout"]))

    model.compile(loss=config["loss"], optimizer=Adam(learning_rate=config["learning_rate"]))
    return model

def save_model_metadata(name, labels, config, history):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO models (name, labels, config, history)
        VALUES (%s, %s, %s, %s)
    """, (name, json.dumps(labels), json.dumps(config), json.dumps(history.history)))
    conn.commit()
    cur.close()
    conn.close()

@app.post("/train")
def train_model(data: TimeSeriesPayload):
    try:
        df = pd.DataFrame(data.values, columns=["timestamp", "value"])
        df["value"] = df["value"].astype(float)

        values = df["value"].values
        window = 10
        X, y = [], []
        for i in range(len(values) - window):
            X.append(values[i:i+window])
            y.append(values[i+window])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Модельная конфигурация (можно динамически доставать по имени из конфига)
        config = {
            "layers": [
                {"type": "LSTM", "units": 64, "return_sequences": True},
                {"type": "LSTM", "units": 32, "return_sequences": False},
                {"type": "Dense", "units": 1}
            ],
            "loss": "mean_squared_error",
            "optimizer": "adam",
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "validation_split": 0.2
        }

        model = build_model((X.shape[1], 1), config)
        history = model.fit(X, y, **{k: config[k] for k in ("epochs", "batch_size", "validation_split")})

        save_model_metadata(data.name, data.labels, config, history)

        return {"message": "Model trained and saved"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))