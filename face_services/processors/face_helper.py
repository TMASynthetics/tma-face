from typing import Tuple, Dict, Any
import numpy as np
import cv2
import numpy
from cv2.typing import Size

from face_services.typing import Template, Frame, Kps, Matrix


TEMPLATES : Dict[Template, numpy.ndarray[Any, Any]] =\
{
	'arcface': numpy.array(
	[
		[ 38.2946, 51.6963 ],
		[ 73.5318, 51.5014 ],
		[ 56.0252, 71.7366 ],
		[ 41.5493, 92.3655 ],
		[ 70.7299, 92.2041 ]
	]),
	'ffhq': numpy.array(
	[
		[ 192.98138, 239.94708 ],
		[ 318.90277, 240.1936 ],
		[ 256.63416, 314.01935 ],
		[ 201.26117, 371.41043 ],
		[ 313.08905, 371.15118 ]
	])
}


def warp_face(temp_frame : Frame, kps : Kps, template : Template, size : Size) -> Tuple[Frame, Matrix]:
	normed_template = TEMPLATES.get(template) * size[1] / size[0]
	affine_matrix = cv2.estimateAffinePartial2D(np.array(kps), normed_template, method = cv2.LMEDS)[0]
	crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (size[1], size[1]))
	return crop_frame, affine_matrix


def paste_back(temp_frame: Frame, crop_frame: Frame, affine_matrix: Matrix) -> Frame:
	inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
	temp_frame_height, temp_frame_width = temp_frame.shape[0:2]
	crop_frame_height, crop_frame_width = crop_frame.shape[0:2]
	inverse_mask = numpy.full((crop_frame_height, crop_frame_width), 255).astype(numpy.float32)
	inverse_crop_frame = cv2.warpAffine(inverse_mask, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
	inverse_temp_frame = cv2.warpAffine(crop_frame, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
	inverse_mask_size = int(numpy.sqrt(numpy.sum(inverse_crop_frame == 255)))
	kernel_size = max(inverse_mask_size // 10, 10)
	inverse_crop_frame = cv2.erode(inverse_crop_frame, numpy.ones((kernel_size, kernel_size)))
	kernel_size = max(inverse_mask_size // 20, 5)
	blur_size = kernel_size * 2 + 1
	inverse_blur_frame = cv2.GaussianBlur(inverse_crop_frame, (blur_size , blur_size), 0) / 255
	inverse_blur_frame = numpy.reshape(inverse_blur_frame, [ temp_frame_height, temp_frame_width, 1 ])
	temp_frame = inverse_blur_frame * inverse_temp_frame + (1 - inverse_blur_frame) * temp_frame
	temp_frame = temp_frame.astype(numpy.uint8)
	return temp_frame

def resize_frame_dimension(frame : Frame, max_width : int, max_height : int) -> Frame:
	height, width = frame.shape[:2]
	if height > max_height or width > max_width:
		scale = min(max_height / height, max_width / width)
		new_width = int(width * scale)
		new_height = int(height * scale)
		return cv2.resize(frame, (new_width, new_height))
	return frame