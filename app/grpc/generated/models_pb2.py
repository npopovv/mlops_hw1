# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: models.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 27, 2, "", "models.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0cmodels.proto\x12\x05model"\x13\n\x03Row\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02"\xb0\x01\n\x0cTrainRequest\x12\x12\n\nmodel_type\x18\x01 \x01(\t\x12/\n\x06params\x18\x02 \x03(\x0b\x32\x1f.model.TrainRequest.ParamsEntry\x12\x1b\n\x07X_train\x18\x03 \x03(\x0b\x32\n.model.Row\x12\x0f\n\x07y_train\x18\x04 \x03(\x05\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01"\x07\n\x05\x45mpty"!\n\rTrainResponse\x12\x10\n\x08model_id\x18\x01 \x01(\x05"-\n\tModelList\x12 \n\x06models\x18\x01 \x03(\x0b\x32\x10.model.ModelInfo"%\n\tModelInfo\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0c\n\x04type\x18\x02 \x01(\t"0\n\x0ePredictRequest\x12\x10\n\x08model_id\x18\x01 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t"%\n\x0fPredictResponse\x12\x12\n\nprediction\x18\x01 \x03(\x02"!\n\rDeleteRequest\x12\x10\n\x08model_id\x18\x01 \x01(\x05" \n\x0e\x44\x65leteResponse\x12\x0e\n\x06status\x18\x01 \x01(\t" \n\x0eStatusResponse\x12\x0e\n\x06status\x18\x01 \x01(\t2\x99\x02\n\x0cModelService\x12\x37\n\nTrainModel\x12\x13.model.TrainRequest\x1a\x14.model.TrainResponse\x12+\n\tGetModels\x12\x0c.model.Empty\x1a\x10.model.ModelList\x12\x38\n\x07Predict\x12\x15.model.PredictRequest\x1a\x16.model.PredictResponse\x12:\n\x0b\x44\x65leteModel\x12\x14.model.DeleteRequest\x1a\x15.model.DeleteResponse\x12-\n\x06Status\x12\x0c.model.Empty\x1a\x15.model.StatusResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "models_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_TRAINREQUEST_PARAMSENTRY"]._loaded_options = None
    _globals["_TRAINREQUEST_PARAMSENTRY"]._serialized_options = b"8\001"
    _globals["_ROW"]._serialized_start = 23
    _globals["_ROW"]._serialized_end = 42
    _globals["_TRAINREQUEST"]._serialized_start = 45
    _globals["_TRAINREQUEST"]._serialized_end = 221
    _globals["_TRAINREQUEST_PARAMSENTRY"]._serialized_start = 176
    _globals["_TRAINREQUEST_PARAMSENTRY"]._serialized_end = 221
    _globals["_EMPTY"]._serialized_start = 223
    _globals["_EMPTY"]._serialized_end = 230
    _globals["_TRAINRESPONSE"]._serialized_start = 232
    _globals["_TRAINRESPONSE"]._serialized_end = 265
    _globals["_MODELLIST"]._serialized_start = 267
    _globals["_MODELLIST"]._serialized_end = 312
    _globals["_MODELINFO"]._serialized_start = 314
    _globals["_MODELINFO"]._serialized_end = 351
    _globals["_PREDICTREQUEST"]._serialized_start = 353
    _globals["_PREDICTREQUEST"]._serialized_end = 401
    _globals["_PREDICTRESPONSE"]._serialized_start = 403
    _globals["_PREDICTRESPONSE"]._serialized_end = 440
    _globals["_DELETEREQUEST"]._serialized_start = 442
    _globals["_DELETEREQUEST"]._serialized_end = 475
    _globals["_DELETERESPONSE"]._serialized_start = 477
    _globals["_DELETERESPONSE"]._serialized_end = 509
    _globals["_STATUSRESPONSE"]._serialized_start = 511
    _globals["_STATUSRESPONSE"]._serialized_end = 543
    _globals["_MODELSERVICE"]._serialized_start = 546
    _globals["_MODELSERVICE"]._serialized_end = 827
# @@protoc_insertion_point(module_scope)
