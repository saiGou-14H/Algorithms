# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: video.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'video.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bvideo.proto\"k\n\nVideoFrame\x12\x12\n\nimage_data\x18\x01 \x01(\x0c\x12\x11\n\ttimestamp\x18\x02 \x01(\x03\x12\x17\n\x0f\x61lgorithms_type\x18\x03 \x01(\x03\x12\x0e\n\x06height\x18\x04 \x01(\x03\x12\r\n\x05width\x18\x05 \x01(\x03\"b\n\x0e\x41nalysisResult\x12\x11\n\ttimestamp\x18\x01 \x01(\x03\x12\x1b\n\x05\x62oxes\x18\x02 \x03(\x0b\x32\x0c.BoundingBox\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\t\x12\x12\n\nimage_data\x18\x04 \x01(\x0c\"\x80\x01\n\x0b\x42oundingBox\x12\r\n\x05label\x18\x01 \x01(\t\x12\t\n\x01x\x18\x02 \x01(\x03\x12\t\n\x01y\x18\x03 \x01(\x03\x12\r\n\x05width\x18\x04 \x01(\x03\x12\x0e\n\x06height\x18\x05 \x01(\x03\x12\r\n\x05score\x18\x06 \x01(\x02\x12\x10\n\x08track_id\x18\x07 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x08 \x01(\t2B\n\x0eVideoProcessor\x12\x30\n\x0cProcessFrame\x12\x0b.VideoFrame\x1a\x0f.AnalysisResult(\x01\x30\x01\x42\x1f\n\x0f\x63om.saigou.grpcB\nVideoProtoP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'video_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\017com.saigou.grpcB\nVideoProtoP\001'
  _globals['_VIDEOFRAME']._serialized_start=15
  _globals['_VIDEOFRAME']._serialized_end=122
  _globals['_ANALYSISRESULT']._serialized_start=124
  _globals['_ANALYSISRESULT']._serialized_end=222
  _globals['_BOUNDINGBOX']._serialized_start=225
  _globals['_BOUNDINGBOX']._serialized_end=353
  _globals['_VIDEOPROCESSOR']._serialized_start=355
  _globals['_VIDEOPROCESSOR']._serialized_end=421
# @@protoc_insertion_point(module_scope)
