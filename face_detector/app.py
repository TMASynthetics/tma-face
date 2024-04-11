import base64
import json
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import os 
import numpy as np
from typing import Annotated, List
from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile

from face_detector.face_detector import FaceDetector


tags_metadata = [
    {
        "name": "Face Services API",
        "description": "This API provides human face processing capabilities.",
        # "externalDocs": {
        #     "description": "Documentation",
        #     "url": "https://jwsite.sharepoint.com/:f:/r/sites/WHQ-MEPS-TMASyntheticMedia-Team/Shared%20Documents/Products/Face%20Services%20API?csf=1&web=1&e=IVOU8p",
        # },
    },
]

app = FastAPI(
    title="TMA - Synthetic Media Team - Face Services API",
    description="",
    version="1.0",
    contact={
        "name": "Thierry SAMMOUR",
        "email": "tsammour@bethel.jw.org",
    },
    openapi_tags=tags_metadata
)

@app.route('/', include_in_schema=False)
def app_redirect(_):
    return RedirectResponse(url='/docs')

class FaceDetectorRequest(BaseModel):
    image_path: str
    model_path: str
    output_path: str

@app.post("/faces_services/face_detector", tags=["Testing"])
async def face_detector(face_detector_request: FaceDetectorRequest):
    face_detector_output = None
    if face_detector_request.image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and os.path.isfile(face_detector_request.image_path):
        face_detector = FaceDetector(model_path=face_detector_request.model_path)
        face_detector_output = face_detector.run(cv2.imread(face_detector_request.image_path))
        json_object = json.dumps(face_detector_output)
        id = face_detector_request.image_path.lower().split('/')[-1].split('.')[0]

        if os.path.isdir(face_detector_request.output_path):
            with open(os.path.join(face_detector_request.output_path, id + ".json"), "w") as outfile:
                outfile.write(json_object)

            return face_detector_output
        else:
            return 'output folder {} does not exist'.format(face_detector_request.output_path)
    else:
        return 'input image {} is not valid'.format(face_detector_request.image_path)


