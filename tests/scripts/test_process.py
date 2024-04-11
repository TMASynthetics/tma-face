import glob
import logging
import os
import subprocess
import sys
import cv2
sys.path.append(os.getcwd())

from face_services.components.video import Video
from face_services.components.audio import Audio
import onnxruntime



model = onnxruntime.InferenceSession('/Users/to104737/Documents/MATLAB/mpvs_landing/models/ydet/ydet.onnx', providers = ['CoreMLExecutionProvider'])
model = onnxruntime.InferenceSession('/Users/to104737/Documents/MATLAB/mpvs_landing/models/yolov8/yolov8n.onnx', providers = ['CoreMLExecutionProvider'])
model = onnxruntime.InferenceSession('/Users/to104737/Documents/MATLAB/mpvs_landing/models/yolonas/yolonas.onnx', providers = ['CoreMLExecutionProvider'])

video = Video('tests/files/video.mov')

Video.extract_and_save_all_frames(video_path=video.path, output_folder='tests/files', fps=None, trim_frame_end=None)

  
subprocess.call(['ffmpeg', "-i", video.path, 
                "-c:v", "prores_ks", '-profile:v', '2', "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                'tests/files/' + video.name + '_reconstructed.mov'])

Video.create_video_from_images(frames_folder_path='tests/files/' + video.name, 
                               output_video_path='tests/files/' + video.name + '_frames_reconstructed.mov', 
                               fps=video.fps)



# # audio_path = video.extract_audio_from_video(video_path=video.path, extracted_audio_folder='tests/files/vd1')

# # video.add_audio_to_video(video_path=video.path, audio_path=audio_path, output_video_path='tests/files/vd1/vd1.mp4')

# video.create_video_from_images('tests/files/vd1', 
#                                output_video_path='tests/files/vd1/vd1_reconstructed.mp4', 
#                                fps=video.fps, audio_path='tests/files/vd_fr.wav')

# # frames = video.get_frames_from_video(1, 10)

# # frames = video.get_frames_from_files(folder='tests/files/vd1', index_start=1, index_end=None, file_extension='png')

print()
