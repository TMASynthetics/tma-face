import logging
import os
from typing import List
import cv2
import numpy as np
import subprocess
from datetime import datetime

class Video:
    def __init__(self, path: str, frame_start: int = -1, frame_stop: int = -1, 
                 time_start_ms: float = -1.0, time_stop_ms: float = -1.0) -> None:
        
        self.path = path
        self._video = cv2.VideoCapture(self.path)

        self.frame_start = frame_start
        if self.frame_start > -1:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)
        self.frame_stop = frame_stop

        self.time_start_ms = time_start_ms
        if self.time_start_ms >= 0:
            self._video.set(cv2.CAP_PROP_POS_MSEC, self.time_start_ms)
        self.time_stop_ms = time_stop_ms
  
    @property    
    def name(self) -> str:
        return self.path.split('/')[-1].split('.')[0]
    
    @property    
    def is_video(self) -> bool:
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    
    @property    
    def fps(self) -> float:
        return self._video.get(cv2.CAP_PROP_FPS)
    
    @property    
    def frame_number(self) -> int:
        return 1 if int(self._video.get(cv2.CAP_PROP_FRAME_COUNT)) < 0 else int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property    
    def width(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property    
    def height(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
          
    @property    
    def duration(self) -> float:
        return self.frame_number / self.fps
    
    def get_frame_position_by_time(self, position_ms) -> np.array:
        self._video.set(cv2.CAP_PROP_POS_MSEC, position_ms)
        return self.get_frame()   
    
    def get_frame_position_by_index(self, position_index) -> np.array:
        self._video.set(cv2.CAP_PROP_POS_FRAMES, position_index)
        return self.get_frame()   
    
    def get_current_frame_position(self) -> int:
        return self._video.get(cv2.CAP_PROP_POS_FRAMES)
    
    def get_current_frame_timestamp(self) -> float:
        return self._video.get(cv2.CAP_PROP_POS_MSEC)
    
    def get_frame(self) -> np.array:
        if self.frame_stop > -1 and self.get_current_frame_position() > self.frame_stop:
            return None
        elif self.time_stop_ms > -1 and self.get_current_frame_timestamp() > self.time_stop_ms:
            return None
        else:
            _, frame = self._video.read()
            return frame
    
    def get_frames_from_video(self, index_start=1, index_end=None) -> List:
        frames = []
        if not index_end:
            index_end = self.frame_number
        for index in range(index_start, index_end):
            frames.append(self.get_frame_position_by_index(index))
        return frames
    
    def get_frames_from_files(self, folder, index_start=1, index_end=None, file_extension: str='png') -> List:
        frames = []
        if not index_end:
            index_end = self.frame_number
        for index in range(index_start, index_end):
            frames.append(cv2.imread(os.path.join(folder, 'frame_{}.{}'.format(index, file_extension))))
        return frames
    
    @staticmethod  
    def extract_and_save_all_frames(video_path, output_folder, fps, file_extension: str='png', 
                                    trim_frame_start: int=None, trim_frame_end: int=None):

        os.makedirs(os.path.join(output_folder, video_path.split('/')[-1].split('.')[0]), exist_ok=True)

        output_frames_path = os.path.join(os.path.join(output_folder, video_path.split('/')[-1].split('.')[0]), 
                                          video_path.split('/')[-1].split('.')[0] + '_%06d.'+ 
                                          Video.check_frame_extension(file_extension))
        
        command = []
        command.extend(['ffmpeg', '-i', video_path, '-q:v', '1', '-colorspace', '1', '-color_primaries', '1', '-color_trc', '1'])
        command.extend(['-vf'])

        output_cmd = ''
        if fps is not None:
            output_cmd += 'fps=' + str(fps) + ','
        else:
            output_cmd += 'fps=' + str(Video.get_video_frame_rate(video_path)) + ','

        if trim_frame_start is not None and trim_frame_end is not None:
            output_cmd += 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end)
        elif trim_frame_start is not None:
            output_cmd += 'trim=start_frame=' + str(trim_frame_start)
        elif trim_frame_end is not None:
            output_cmd += 'trim=end_frame=' + str(trim_frame_end)
        if len(output_cmd) > 0:
            command.extend([output_cmd])

        command.extend([output_frames_path])
        
        subprocess.call(command)

    @staticmethod  
    def get_video_frame_rate(filename):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                "-show_entries",
                "stream=r_frame_rate",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        result_string = result.stdout.decode('utf-8').split()[0].split('/')
        fps = float(result_string[0])/float(result_string[1])
        return fps
        
    @staticmethod    
    def check_frame_extension(file_extension) -> int:
        return file_extension if file_extension in ['png', 'jpg', 'bmp'] else 'png'

    @staticmethod    
    def extract_audio_from_video(video_path, extracted_audio_folder):
        output_audio_file = os.path.join(extracted_audio_folder, video_path.split('/')[-1].split('.')[0] + '.wav')
        subprocess.call(['ffmpeg', "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ac", "2", output_audio_file])
        return output_audio_file

    @staticmethod   
    def add_audio_to_video(video_path, audio_path, output_video_path):
        subprocess.call(['ffmpeg', "-y", "-i", video_path, "-i",audio_path, "-c:v", "copy", "-strict",
                   "experimental", output_video_path])
        return output_video_path
      
    @staticmethod  
    def create_video_from_images(frames_folder_path, output_video_path, fps, audio_path=None):

        if audio_path:
            subprocess.call(['ffmpeg', "-y", "-framerate", str(fps), "-start_number", "0", 
                            "-pattern_type", "glob", "-i", os.path.join(frames_folder_path, '*.png'),
                            "-i", audio_path, "-map", "0:0", "-map", "1:a", 
                            "-c:v", "prores_ks", '-profile:v', '2', "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                            '-colorspace', 'bt709', '-color_primaries', '1', '-color_trc', '1',
                            output_video_path])
        else:
            subprocess.call(['ffmpeg', "-y", "-framerate", str(fps), "-start_number", "0", 
                            "-pattern_type", "glob", "-i", os.path.join(frames_folder_path, '*.png'), 
                            "-map", "0:0", 
                            "-c:v", "prores_ks", '-profile:v', '2', "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                            '-colorspace', 'bt709', '-color_primaries', '1', '-color_trc', '1',
                            output_video_path])
                 
        return output_video_path
    

    # @staticmethod   
    # def extract_and_save_sample(video_path, output_video_path, time_start_ms: float = 0, time_stop_ms: float = 0):
    #     time_start_str = datetime.utcfromtimestamp(time_start_ms//1000).replace(microsecond=time_start_ms%1000*1000).strftime('%T.%f')[:-3]
    #     time_stop_str = datetime.utcfromtimestamp(time_stop_ms//1000).replace(microsecond=time_stop_ms%1000*1000).strftime('%T.%f')[:-3]
    #     subprocess.call(['ffmpeg', "-i", video_path, "-ss", time_start_str, "-to", time_stop_str, "-async", "1", output_video_path, "-y"])
    #     return output_video_path