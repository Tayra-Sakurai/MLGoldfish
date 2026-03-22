import tensorflow as tf
from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("yolo26n.pt")

result = model.train(data="lvis.yaml", epochs=100, imgsz=640)

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
