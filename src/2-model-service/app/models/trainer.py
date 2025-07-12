import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import List, Dict, Tuple  # Добавлен импорт Tuple

class ModelTrainer:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def prepare_data(self, values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(values) - self.window_size):
            X.append(values[i:i+self.window_size])
            y.append(values[i+self.window_size])
        return np.array(X), np.array(y)

    def build_model(self, config: Dict) -> Sequential:
        model = Sequential()
        
        for i, layer in enumerate(config["layers"]):
            if layer["type"] == "LSTM":
                if i == 0:
                    model.add(LSTM(
                        layer["units"],
                        return_sequences=layer.get("return_sequences", False),
                        input_shape=(self.window_size, 1)
                    ))
                else:
                    model.add(LSTM(
                        layer["units"],
                        return_sequences=layer.get("return_sequences", False)
                    ))
            elif layer["type"] == "Dense":
                model.add(Dense(layer["units"]))
            
            if config.get("dropout") and layer.get("dropout_after", True):
                model.add(Dropout(config["dropout"]))

        model.compile(
            loss=config["loss"],
            optimizer=Adam(learning_rate=config.get("learning_rate", 0.001))
        )
        return model