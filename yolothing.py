import cv2 as cv
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
modelis = YOLO("yolo11x.pt")
vCap = cv.VideoCapture("C:\\Users\\liela\\Downloads\\video.mp4.mp4")

fonts = {
    20: ImageFont.truetype("C:\\yolothingy\\DejaVuSans.ttf", 20),
    30: ImageFont.truetype("C:\\yolothingy\\DejaVuSans.ttf", 30),
    40: ImageFont.truetype("C:\\yolothingy\\DejaVuSans.ttf", 40)
}

def utf8Text(frame, text, pos, size, color):
    imgPil = Image.fromarray(frame)
    draw = ImageDraw.Draw(imgPil)
    draw.text(pos, text, font=fonts[size], fill=color)
    return np.array(imgPil)

stavvietas = [
    (405, 478, 560, 551),
    (565, 473, 718, 540),
    (720, 470, 865, 534),
    (238, 481, 428, 558),
    (62, 489, 280, 566),
    (871, 473, 1018, 532),
    (1018, 463, 1163, 527),
]

laikaUzskaite = {i: None for i in range(len(stavvietas))}
tuksiKadri = {i: 0 for i in range(len(stavvietas))}
buffers = 2

def parklajums(auto, vieta):
    xA = max(auto[0], vieta[0])
    yA = max(auto[1], vieta[1])
    xB = min(auto[2], vieta[2])
    yB = min(auto[3], vieta[3])

    if xB < xA or yB < yA:
        return 0

    inter = (xB - xA) * (yB - yA)
    vietaLauk = (vieta[2] - vieta[0]) * (vieta[3] - vieta[1])
    return inter / vietaLauk

while True:
    ret, frame = vCap.read()
    if not ret:
        break
    frame = cv.resize(frame, (1280, 720))
    rezultats = modelis(frame, conf=0.15, device=device)
    res = rezultats[0]
    masinas = []

    for kaste in res.boxes:
        cls = int(kaste.cls[0])
        tKlases = [2, 3, 5, 7]
        if cls in tKlases:
            x1, y1, x2, y2 = map(int, kaste.xyxy[0])
            masinas.append((x1, y1, x2, y2))
            nosaukums = "Transportlīdzeklis"
            if cls == 2:
                nosaukums = "Mašīna"
            elif cls == 3:
                nosaukums = "Divritenis"
            elif cls == 5:
                nosaukums = "Autobuss"
            elif cls == 7:
                nosaukums = "Kravas auto"
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = utf8Text(frame, nosaukums, (x1, y1 - 25), 20, (0, 255, 0))

    brivasVietas = 0

    for i, vieta in enumerate(stavvietas):
        paslaikAiznemts = False
        for tLidzeklis in masinas:
            if parklajums(tLidzeklis, vieta) > 0.3:
                paslaikAiznemts = True
                break
        x1, y1, x2, y2 = vieta
        if paslaikAiznemts:
            tuksiKadri[i] = 0
            if laikaUzskaite[i] is None:
                laikaUzskaite[i] = time.time()
        else:
            tuksiKadri[i] += 1
        if laikaUzskaite[i] is not None and tuksiKadri[i] < buffers:
            krasa = (0, 0, 255)
            pavaditaisLaiks = int(time.time() - laikaUzskaite[i])
            frame = utf8Text(frame, f"{pavaditaisLaiks}s", (x1, y1 - 25), 20, (0, 0, 255))
        else:
            krasa = (0, 255, 0)
            brivasVietas += 1
            laikaUzskaite[i] = None

        cv.rectangle(frame, (x1, y1), (x2, y2), krasa, 2)

    frame = utf8Text(frame, f"Brīvas vietas: {brivasVietas}/{len(stavvietas)}", (50, 40), 30, (255, 255, 255))
    cv.imshow("Stāvvietas detektēšana", frame)
    if cv.waitKey(1) & 0xFF == 27: #ESC poga
        break

vCap.release()
cv.destroyAllWindows()