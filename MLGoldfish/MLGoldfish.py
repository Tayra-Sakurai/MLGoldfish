from ultralytics import YOLO
import cv2
from pathlib import Path
from PIL import Image
from numpy import ndarray
from tkinter.filedialog import askdirectory
import os

model = YOLO("yolov8n-oiv7.pt")

image_path = Path(askdirectory(title="Data Source"))

i: int = 0

for path in image_path.glob("**/*.jpg"):
    cv = cv2.imread(str(path))
    print(f'Image "{path.name}" was loaded.')
    if cv is None:
        continue
    resultL = model(cv, save_dir="result")
    for result in resultL:
        i += 1
        im_bgr: ndarray = result.plot()
        im_rbg = Image.fromarray(im_bgr[..., ::-1])

        Path("plot").mkdir(exist_ok=True)

        im_rbg.save(f"plot/plot ({i}).png")

        p = Path("result")
        p.mkdir(exist_ok=True)

        result.save()

os.system.pause()
