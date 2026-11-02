"""YOLOv8n training. Owner: Krishna Anand.
Final run: python train/train_yolov8n.py --epochs 100 --name yolov8n_100
Metrics land in runs/<name>/results.csv; the comparison reads exported predictions instead."""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--name", default="yolov8n_100")
a = ap.parse_args()

m = YOLO("yolov8n.pt")
m.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, seed=0,
        close_mosaic=10, project="runs", name=a.name, exist_ok=True, plots=True)
r = m.val(data=a.data, split="val", imgsz=640)
print(f"mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} P={r.box.mp:.4f} R={r.box.mr:.4f}")
