import grpc
from concurrent import futures
import logging
import random
import joblib
from app.models import ModelManager
import app.models_pb2 as model_pb2
import app.models_pb2_grpc as model_pb2_grpc

#для генерации файлов используем команду
#python -m grpc_tools.protoc -I./app --python_out=./app --grpc_python_out=./app app/model.proto


# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelServiceServicer(model_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        self.model_manager = ModelManager()

    def TrainModel(self, request, context):
        logger.info(f"gRPC call to train model of type {request.model_type} with params: {request.params}")
        try:
            model_id = self.model_manager.train(request.model_type, request.params)
            return model_pb2.TrainResponse(model_id=model_id)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_pb2.TrainResponse()

    def GetModels(self, request, context):
        logger.info("gRPC call to get list of models")
        models = self.model_manager.get_available_models()
        model_list = [model_pb2.ModelInfo(id=m["id"], type=m["type"]) for m in models]
        return model_pb2.ModelList(models=model_list)

    def Predict(self, request, context):
        logger.info(f"gRPC call to predict with model ID: {request.model_id} and data: {request.data}")
        try:
            data = list(map(float, request.data.split()))
            prediction = self.model_manager.predict(request.model_id, data)
            return model_pb2.PredictResponse(prediction=prediction)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_pb2.PredictResponse()

    def DeleteModel(self, request, context):
        logger.info(f"gRPC call to delete model with ID: {request.model_id}")
        try:
            self.model_manager.delete(request.model_id)
            return model_pb2.DeleteResponse(status="deleted")
        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_pb2.DeleteResponse()

    def Status(self, request, context):
        logger.info("gRPC call to get status")
        return model_pb2.StatusResponse(status="running")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServiceServicer_to_server(ModelServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
