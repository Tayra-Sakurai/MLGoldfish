from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("yolov8n-oiv7.pt")

image_path = Path("C:\\Users\\taira\\Pictures\\カメラ ロール\\Goldfish")

for path in image_path.glob("**/*.jpg"):
    cv = cv2.imread(str(path))
    print(f'Image "{path.name}" was loaded.')
    if cv is None:
        continue
    resultL = model(cv, save_dir="result")
    for result in resultL:
        result.save_crop("C:\\Users\\taira\\source\\repos\\MLGoldfish\\MLGoldfish\\result")
        print("Saved!")
