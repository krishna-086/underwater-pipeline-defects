"""YOLOv8n preliminary run. Owner: Krishna Anand.
Metrics: last row of runs/yolov8n_prelim/results.csv, or the final print below."""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

m = YOLO("yolov8n.pt")
m.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, seed=0,
        project="runs", name="yolov8n_prelim", exist_ok=True, plots=True)
r = m.val(data=a.data, split="val", imgsz=640)
print(f"mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} P={r.box.mp:.4f} R={r.box.mr:.4f}")
