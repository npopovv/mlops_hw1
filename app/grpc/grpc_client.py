import grpc
import sys
import os
sys.path.insert(0, '/home/nikita/mlops_hw1/app/grpc/generated')
import models_pb2
import models_pb2_grpc
from google.protobuf.struct_pb2 import Struct

def train_model(stub, model_type, params, X_train, y_train):

    # Заполняем запрос
    request = models_pb2.TrainRequest(
        model_type=model_type,
        params=params,
        X_train=[models_pb2.Row(data=row) for row in X_train],
        y_train=y_train
    )
    response = stub.TrainModel(request)
    print("TrainModel response:", response)
    return response.model_id

def get_models(stub):
    request = models_pb2.Empty()
    response = stub.GetModels(request)
    print("GetModels response:", response)
    return response.models

def predict(stub, model_id, data):
    request = models_pb2.PredictRequest(model_id=model_id, data=data)
    response = stub.Predict(request)
    print("Predict response:", response)
    return response.prediction

def delete_model(stub, model_id):
    request = models_pb2.DeleteRequest(model_id=model_id)
    response = stub.DeleteModel(request)
    print("DeleteModel response:", response)
    return response.status

def main():
    # Создаем канал и подключаемся к gRPC-серверу
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = models_pb2_grpc.ModelServiceStub(channel)
        
        # Пример использования методов
        print("Training a logistic model...")
        # Пример двумерного массива данных
        X_train = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        y_train = [0, 1, 0]
        model_id = train_model(stub, "logistic", {}, X_train, y_train)
        
        print("\nListing all models...")
        models = get_models(stub)
        
        print("\nMaking a prediction with the trained model...")
        prediction = predict(stub, model_id, '0.1 0.2 0.3')
        
        print("\nDeleting the model...")
        delete_status = delete_model(stub, model_id)

if __name__ == "__main__":
    main()

