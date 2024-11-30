import grpc
from concurrent import futures
from app.logging_config import setup_logger
import os
import sys
from app.models import ModelManager 

sys.path.insert(0, '/home/nikita/mlops_hw1/app/grpc/generated')
import app.grpc.generated.models_pb2 as models_pb2
import app.grpc.generated.models_pb2_grpc as models_pb2_grpc

logger = setup_logger(__name__)

class ModelServiceServicer(models_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        # Инициализация менеджера моделей
        self.model_manager = ModelManager()

    def TrainModel(self, request, context):
        logger.info(f"gRPC call to train model of type {request.model_type} with params: {request.params}")
        try:
            model_type = request.model_type
            params = dict(request.params)

            # Десериализация данных
            X_train = [list(row.data) for row in request.X_train]
            y_train = list(request.y_train)

            model_id = self.model_manager.train(model_type, params, X_train, y_train)
            return models_pb2.TrainResponse(model_id=model_id)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return models_pb2.TrainResponse()


    def GetModels(self, request, context):
        logger.info("gRPC call to get list of models")
        try:
            models = self.model_manager.get_available_models()
            model_list = [models_pb2.ModelInfo(id=m['id'], type="unknown") for m in models]
            return models_pb2.ModelList(models=model_list)
        except Exception as e:
            logger.error(f"GetModels failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return models_pb2.ModelList()

    def Predict(self, request, context):
        logger.info(f"gRPC call to predict with model ID: {request.model_id}")
        try:
            data = list(map(float, request.data.split()))
            prediction = self.model_manager.predict(request.model_id, data)
            return models_pb2.PredictResponse(prediction=prediction)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return models_pb2.PredictResponse()

    def DeleteModel(self, request, context):
        logger.info(f"gRPC call to delete model with ID: {request.model_id}")
        try:
            self.model_manager.delete(request.model_id)
            return models_pb2.DeleteResponse(status="deleted")
        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return models_pb2.DeleteResponse()

    def Status(self, request, context):
        logger.info("gRPC call to get status")
        return models_pb2.StatusResponse(status="running")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    models_pb2_grpc.add_ModelServiceServicer_to_server(ModelServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("Starting gRPC server on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

