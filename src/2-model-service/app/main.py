from fastapi import FastAPI, HTTPException
from .models.database import DatabaseManager
from .models.trainer import ModelTrainer
from .schemas import TimeSeriesData, ModelConfig, ModelResponse
import os

app = FastAPI()

# Configuration
DB_DSN = os.getenv("DB_DSN", "postgresql://mad:secretPASSW0rd@db/ml_models")

# Initialize components
db_manager = DatabaseManager(DB_DSN)
model_trainer = ModelTrainer()

@app.post("/models", response_model=ModelResponse)
async def train_model(data: TimeSeriesData, config: ModelConfig):
    try:
        # Prepare data
        values = [v[1] for v in data.values]  # Extract values from [timestamp, value]
        X, y = model_trainer.prepare_data(values)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        model = model_trainer.build_model(config.dict())
        history = model.fit(
            X, y,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split
        )
        
        # Save model metadata
        model_id = db_manager.save_model(
            data.name,
            data.labels,
            config.dict(),
            history.history
        )
        
        return ModelResponse(
            id=str(model_id),
            name=data.name,
            status="success",
            loss=history.history["loss"][-1],
            val_loss=history.history.get("val_loss", [None])[-1]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    model = db_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model