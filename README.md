Face Services API
==========
> State-of-the-art human face processing.
> - [x] Face detection and analysis

Installation
------------
Please use Python 3.10.

1. Install the requirements:
```
pip install -r requirements.txt
```

Usage
-----
Run the command:
```
uvicorn face_detector.app:app --host 0.0.0.0 --port 80 
```

Docker
-----
```
docker build . -t face_detector
docker run -d --name face_detector_container -p 80:80 face_detector
```
