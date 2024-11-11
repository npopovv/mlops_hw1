from enum import Enum
import random
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

class ModelType(str, Enum):
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"

class ModelManager:
    def __init__(self):
        self.models = {}
        self.next_id = 1

    def train(self, model_type: ModelType, params: dict):
        logger.info(f"Starting training for model type: {model_type}")
        
        if model_type == ModelType.LOGISTIC:
            model = LogisticRegression(**params)
        elif model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(**params)
        else:
            logger.error("Unknown model type")
            raise ValueError("Unknown model type")

        model.fit([[random.random() for _ in range(5)] for _ in range(100)], [random.randint(0, 1) for _ in range(100)])
        model_id = self.next_id
        self.models[model_id] = model
        self.next_id += 1
        joblib.dump(model, f"model_{model_id}.pkl")

        logger.info(f"Model trained and saved with ID: {model_id}")
        return model_id

    def get_available_models(self):
        logger.info("Fetching list of available models")
        return [{"id": model_id, "type": type(model).__name__} for model_id, model in self.models.items()]

    def predict(self, model_id: int, data):
        logger.info(f"Received data for prediction with model ID: {model_id}")
        
        if model_id not in self.models:
            logger.error("Model not found")
            raise ValueError("Model not found")

        model = self.models[model_id]
        prediction = model.predict([data]).tolist()
        
        logger.info(f"Prediction result: {prediction}")
        return prediction

    def delete(self, model_id: int):
        logger.info(f"Attempting to delete model with ID: {model_id}")
        
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Model {model_id} deleted successfully")
            return True
        else:
            logger.error("Model not found")
            raise ValueError("Model not found")
