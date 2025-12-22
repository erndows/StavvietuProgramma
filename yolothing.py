import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolo11m-obb.pt")
vcap = cv.VideoCapture("C:\\Users\\liela\\Downloads\\video.mp4.mp4")

while(1):

    ret, frame = vcap.read()
    key = cv.waitKey(1) & 0xFF
    if key == 27: #ESC key
        break
    frame = cv.resize(frame, (1280, 720))
    results = model(frame, conf = 0.1, device="cpu")
    annotated = results[0].plot(labels=False)
    cv.imshow('YOLO LIVE', annotated)
    