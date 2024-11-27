import grpc
import app.models_pb2 as model_pb2
import app.models_pb2_grpc as model_pb2_grpc
from google.protobuf.struct_pb2 import Struct

def train_model(stub, model_type, params):
    # Преобразуем params в Struct
    params_struct = Struct()
    for key, value in params.items():
        params_struct[key] = value

    request = model_pb2.TrainRequest(model_type=model_type, params=params_struct)
    response = stub.TrainModel(request)
    print("TrainModel response:", response)
    return response.model_id

def get_models(stub):
    request = model_pb2.GetModelsRequest()
    response = stub.GetModels(request)
    print("GetModels response:", response)
    return response.models

def predict(stub, model_id, data):
    request = model_pb2.PredictRequest(model_id=model_id, data=data)
    response = stub.Predict(request)
    print("Predict response:", response)
    return response.prediction

def delete_model(stub, model_id):
    request = model_pb2.DeleteRequest(model_id=model_id)
    response = stub.DeleteModel(request)
    print("DeleteModel response:", response)
    return response.status

def main():
    # Создаем канал и подключаемся к gRPC-серверу
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_pb2_grpc.ModelServiceStub(channel)
        
        # Пример использования методов
        print("Training a logistic model...")
        model_id = train_model(stub, "logistic", {})
        
        print("\nListing all models...")
        models = get_models(stub)
        
        print("\nMaking a prediction with the trained model...")
        prediction = predict(stub, model_id, '0.1 0.2 0.3 0.4 0.5')
        
        print("\nDeleting the model...")
        delete_status = delete_model(stub, model_id)

if __name__ == "__main__":
    main()