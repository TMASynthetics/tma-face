import base64
import json
import time
import cv2
import os 
import numpy as np
import onnx
from face_detector import FaceDetector
import onnxruntime
import multiprocessing

face_detector = FaceDetector()


frame = cv2.imread('face_detector/image.jpg')

output = face_detector.run(frame)


model_onnx = onnx.load(face_detector.onnx_path)
model_serialized = model_onnx.SerializeToString()

# model_onnx.SerializeToString()

start_time = time.time()
session = onnxruntime.InferenceSession(face_detector.onnx_path, providers = ['CPUExecutionProvider'])    
elapsed_time = time.time() - start_time
print("Session loaded from onnx file : ", elapsed_time, 's') 


start_time = time.time()
session = onnxruntime.InferenceSession(model_serialized, providers = ['CPUExecutionProvider'])    
elapsed_time = time.time() - start_time
print("Session loaded from onnx string : ", elapsed_time, 's') 




start_time = time.time()
output = face_detector.run(frame)
elapsed_time = time.time() - start_time
print("Inference time : ", elapsed_time, 's') 




n_threads = min(4, multiprocessing.cpu_count() - 1)
print(f"n_threads={n_threads}")


print
