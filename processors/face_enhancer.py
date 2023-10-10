from typing import Any, Optional, List
import threading
import insightface
import numpy
from gfpgan.utils import GFPGANer
from processors.face_analyser import FaceAnalyser
import os
import gfpgan

class FaceEnhancer:
  
	def __init__(self):
		self.model = gfpgan.GFPGANer(model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../.assets/models/GFPGANv1.4.pth')), upscale = 1)
		self.face_analyser = FaceAnalyser()

	def run(self, frame):

		face = self.face_analyser.run(frame)[0]

		start_x, start_y, end_x, end_y = map(int, face['bbox'])
		padding_x = int((end_x - start_x) * 0.5)
		padding_y = int((end_y - start_y) * 0.5)
		start_x = max(0, start_x - padding_x)
		start_y = max(0, start_y - padding_y)
		end_x = max(0, end_x + padding_x)
		end_y = max(0, end_y + padding_y)
		crop_frame = frame[start_y:end_y, start_x:end_x]

		_, _, crop_frame = self.model.enhance(
			crop_frame,
			paste_back = True
		)

		frame[start_y:end_y, start_x:end_x] = crop_frame
	
		return frame





# import os
# import cv2
# import torch
# import gfpgan
# import gdown
# from PIL import Image
# from upscaler.RealESRGAN import RealESRGAN


# def gfpgan_runner(img, model):
#     _, imgs, _ = model.enhance(img, paste_back=True, has_aligned=True)
#     return imgs[0]


# def realesrgan_runner(img, model):
#     img = model.predict(img)
#     return img


# supported_enhancers = {
#     "GFPGAN": ("./pretrained_models/GFPGANv1.4.pth", gfpgan_runner),
#     "REAL-ESRGAN 2x": ("./pretrained_models/RealESRGAN_x2.pth", realesrgan_runner),
#     "REAL-ESRGAN 4x": ("./pretrained_models/RealESRGAN_x4.pth", realesrgan_runner),
#     "REAL-ESRGAN 8x": ("./pretrained_models/RealESRGAN_x8.pth", realesrgan_runner)
# }

# cv2_interpolations = ["LANCZOS4", "CUBIC", "NEAREST"]

# def model_check(model_url, model_path):
#     if not os.path.exists(model_path):
#         gdown.download(model_url, model_path, quiet=False)


# def load_face_enhancer_model(name='GFPGAN', device="cpu"):
#     if name in supported_enhancers.keys():
#         model_path, model_runner = supported_enhancers.get(name)
#         model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
#     if name == 'GFPGAN':
#         model_url = 'https://drive.google.com/uc?id=1QsJPgvZNwFsBktbeYENVsEq663UgBQRj'  
#         model_check(model_url, model_path)
#         model = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=device)
#     elif name == 'REAL-ESRGAN 2x':
#         model_url = 'https://drive.google.com/uc?id=1BYFc4ttYGHmA-GZMmgXW9NdgPkXkgjtv'  
#         model_check(model_url, model_path)
#         model = RealESRGAN(device, scale=2)
#         model.load_weights(model_path, download=False)
#     elif name == 'REAL-ESRGAN 4x':
#         model_url = 'https://drive.google.com/uc?id=1N4MNjfGhrz-CHq99WCp6NEfgzMIGxAE0'  
#         model_check(model_url, model_path)
#         model = RealESRGAN(device, scale=4)
#         model.load_weights(model_path, download=False)
#     elif name == 'REAL-ESRGAN 8x':
#         model_url = 'https://drive.google.com/uc?id=14FtSjtgtl8iySVrrvFDX-HxCCkdbsoPh'  
#         model_check(model_url, model_path)
#         model = RealESRGAN(device, scale=8)
#         model.load_weights(model_path, download=False)
#     elif name == 'LANCZOS4':
#         model = None
#         model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_LANCZOS4)
#     elif name == 'CUBIC':
#         model = None
#         model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
#     elif name == 'NEAREST':
#         model = None
#         model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
#     else:
#         model = None
#     return (model, model_runner)







