from enum import Enum
import os
import joblib
from app.logging_config import setup_logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# настроенный логгер
logger = setup_logger(name=__name__)


class ModelType(str, Enum):
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"


class ModelManager:
    def __init__(self, storage_dir="models_storage"):

        self.storage_dir = storage_dir
        os.makedirs(
            self.storage_dir, exist_ok=True
        )  # если нет папки для пиклов - создаем

    def train(self, model_type: ModelType, params: dict, X_train, y_train):
        logger.info(f"Starting training for model type: {model_type}")

        if model_type == ModelType.LOGISTIC:
            model = LogisticRegression(**params)
        elif model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(**params)
        else:
            logger.error("Unknown model type")
            raise ValueError("Unknown model type")

        model.fit(X_train, y_train)
        # новый индекс модели
        model_id = self._get_next_model_id()

        path_to_save = os.path.join(self.storage_dir, f"model_{model_id}.pkl")
        joblib.dump(model, path_to_save)

        logger.info(f"Model trained and saved with ID: {model_id}")
        return model_id

    def get_available_models(self):
        logger.info("Fetching list of available models")

        model_files = [f for f in os.listdir(self.storage_dir) if f.endswith(".pkl")]
        models = [
            {"id": int(f.split("_")[1].split(".")[0]), "path": f} for f in model_files
        ]
        return sorted(models, key=lambda x: x["id"])

    def predict(self, model_id: int, data):
        logger.info(f"Received data for prediction with model ID: {model_id}")

        model_path = os.path.join(self.storage_dir, f"model_{model_id}.pkl")
        if not os.path.exists(model_path):
            logger.error("Model not found")
            raise ValueError("Model not found")

        model = joblib.load(model_path)

        prediction = model.predict([data]).tolist()

        logger.info(f"Prediction result: {prediction}")
        return prediction

    def delete(self, model_id: int):
        logger.info(f"Attempting to delete model with ID: {model_id}")
        # Удаление модели
        model_path = os.path.join(self.storage_dir, f"model_{model_id}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Model {model_id} deleted successfully")
            return True
        else:
            logger.error("Model not found")
            raise ValueError(f"Model with ID {model_id} not found")

    def _get_next_model_id(self):
        # определяем следующий id
        existing_models = self.get_available_models()
        if not existing_models:
            return 1
        return max(model["id"] for model in existing_models) + 1
