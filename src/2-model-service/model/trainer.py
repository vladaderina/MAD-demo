from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np

async def train_lstm_model(dataframe, config):
    # Преобразуем в последовательности
    window_size = config["data"]["window_size"]
    X, y = [], []
    values = dataframe["value"].values
    for i in range(len(values) - window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size])
    X = np.array(X).reshape((-1, window_size, 1))
    y = np.array(y)

    # Создание модели
    model = Sequential()
    for layer in config["models"][0]["layers"]:
        if layer["type"] == "LSTM":
            model.add(LSTM(layer["units"], return_sequences=layer.get("return_sequences", False)))
        elif layer["type"] == "Dense":
            model.add(Dense(layer["units"]))
    if config["models"][0]["dropout"]:
        model.add(Dropout(config["models"][0]["dropout"]))

    optimizer = Adam(learning_rate=config["models"][0]["learning_rate"])
    model.compile(loss=config["models"][0]["loss"], optimizer=optimizer)

    model.fit(X, y, epochs=config["models"][0]["epochs"], batch_size=config["models"][0]["batch_size"], validation_split=0.2)

    return model, config["models"][0]
