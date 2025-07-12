from pydantic import BaseModel
from typing import List, Dict

class TimeSeriesData(BaseModel):
    name: str
    labels: Dict[str, str]
    values: List[List[float]]  # [[timestamp, value], ...]

class ModelConfig(BaseModel):
    layers: List[Dict]
    loss: str
    optimizer: str = "adam"
    dropout: float = None
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2

class ModelResponse(BaseModel):
    id: str
    name: str
    status: str
    loss: float
    val_loss: float = None