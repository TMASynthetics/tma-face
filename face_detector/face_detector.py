from collections import OrderedDict
from typing import Any, Dict, Optional, List
import uuid
import onnxruntime
import cv2
import numpy as np


def apply_nms(bounding_box_list : List, iou_threshold : float) -> List[int]:
	keep_indices = []
	dimension_list = np.reshape(bounding_box_list, (-1, 4))
	x1 = dimension_list[:, 0]
	y1 = dimension_list[:, 1]
	x2 = dimension_list[:, 2]
	y2 = dimension_list[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	indices = np.arange(len(bounding_box_list))
	while indices.size > 0:
		index = indices[0]
		remain_indices = indices[1:]
		keep_indices.append(index)
		xx1 = np.maximum(x1[index], x1[remain_indices])
		yy1 = np.maximum(y1[index], y1[remain_indices])
		xx2 = np.minimum(x2[index], x2[remain_indices])
		yy2 = np.minimum(y2[index], y2[remain_indices])
		width = np.maximum(0, xx2 - xx1 + 1)
		height = np.maximum(0, yy2 - yy1 + 1)
		iou = width * height / (areas[index] + areas[remain_indices] - width * height)
		indices = indices[np.where(iou <= iou_threshold)[0] + 1]
	return keep_indices

FACE_DETECTOR_OUTPUT = {
	'bbox': [],
	'face_landmark_5': [],
	'score': [],
}

FACE_DETECTOR_MODELS = 	{'detection' :
		{
			'yoloface' :
			{
				'url' : '',
				'path' : 'face_detector/yoloface_8n.onnx',
			},
		}
	}

class FaceDetector:
	def __init__(self):
		self.id = uuid.uuid4()
		self.model = onnxruntime.InferenceSession(FACE_DETECTOR_MODELS['detection']['yoloface']['path'], providers = ['CPUExecutionProvider'])
		self.onnx_path = FACE_DETECTOR_MODELS['detection']['yoloface']['path']

	def run(self, frame):
		face_detector_width, face_detector_height = (640, 640)

		height, width = frame.shape[:2]
		max_width, max_height = (face_detector_width, face_detector_height)

		if height > max_height or width > max_width:
			scale = min(max_height / height, max_width / width)
			new_width = int(width * scale)
			new_height = int(height * scale)
			
			frame = cv2.resize(frame, (new_width, new_height))
	
		ratio_height = frame.shape[0] / frame.shape[0]
		ratio_width = frame.shape[1] / frame.shape[1]
		bounding_box_list = []
		face_landmark_5_list = []
		score_list = []

		detect_vision_frame = np.zeros((face_detector_height, face_detector_width, 3))
		detect_vision_frame[:frame.shape[0], :frame.shape[1], :] = frame
		detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
		detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)

		detections = self.model.run(None,
			{
				self.model.get_inputs()[0].name: detect_vision_frame
			})
		detections = np.squeeze(detections).T
		bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detections, [ 4, 5 ], axis = 1)
		keep_indices = np.where(score_raw > 0.4)[0]
		if keep_indices.any():
			bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
			for bounding_box in bounding_box_raw:
				bounding_box_list.append(np.array(
				[
					(bounding_box[0] - bounding_box[2] / 2) * ratio_width,
					(bounding_box[1] - bounding_box[3] / 2) * ratio_height,
					(bounding_box[0] + bounding_box[2] / 2) * ratio_width,
					(bounding_box[1] + bounding_box[3] / 2) * ratio_height
				]))
			face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
			face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
			for face_landmark_5 in face_landmark_5_raw:
				face_landmark_5_list.append(np.array(face_landmark_5.reshape(-1, 3)[:, :2]))
			score_list = score_raw.ravel().tolist()

		idx = apply_nms(bounding_box_list, 0.4)

		FACE_DETECTOR_OUTPUT['bbox'] = [bounding_box_list[i].tolist() for i in idx]
		FACE_DETECTOR_OUTPUT['face_landmark_5'] = [face_landmark_5_list[i].tolist() for i in idx]
		FACE_DETECTOR_OUTPUT['score'] = [score_list[i] for i in idx]

		return FACE_DETECTOR_OUTPUT






