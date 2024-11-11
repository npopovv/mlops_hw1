from fastapi import FastAPI, HTTPException
from app.models import ModelManager, ModelType
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Настройка логгера
logging.basicConfig(
    level=logging.INFO,  # Уровень логирования
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Формат сообщения
    handlers=[
        logging.FileHandler("app.log"),  # Запись логов в файл app.log
        logging.StreamHandler()  # Вывод логов также в консоль
    ]
)
logger = logging.getLogger(__name__)  # Инициализация логгера для текущего модуля
app = FastAPI()
model_manager = ModelManager()

@app.post("/train/")
async def train_model(model_type: ModelType, params: dict):
    logger.info(f"API call to train model of type {model_type} with params: {params}")
    
    try:
        model_id = model_manager.train(model_type, params)
        logger.info(f"Model trained with ID: {model_id}")
        return {"model_id": model_id}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/")
async def get_models():
    logger.info("API call to get list of models")
    return model_manager.get_available_models()

@app.post("/predict/")
async def predict(model_id: int, data: str):

    data = list(map(float, data.split(' '))) # меняем на норм формат для модели
    logger.info(f"API call to predict with model ID: {model_id} and data: {data}")
    
    try:
        prediction = model_manager.predict(model_id, data)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/")
async def delete_model(model_id: int):
    logger.info(f"API call to delete model with ID: {model_id}")
    
    try:
        model_manager.delete(model_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/")
async def status():
    logger.info("API call to get status")
    return {"status": "running"}
