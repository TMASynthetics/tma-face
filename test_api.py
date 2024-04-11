import uvicorn
# from face_services.app import app
from face_detector.app import app
uvicorn.run(app, host='0.0.0.0', port=80)



{
  "image_path": "/Users/to104737/Documents/DEV/PYTHON/JW/tma-face-services/face_detector/image.jpg",
  "output_path": "/Users/to104737/Documents/DEV/PYTHON/JW/tma-face-services/face_detector"
}