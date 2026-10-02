"""RT-DETR-L preliminary run. Owner: Harish Bharathwaj.
batch 8 and AdamW because the DETR head is heavier and unstable with SGD defaults."""
import argparse
from ultralytics import RTDETR

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

m = RTDETR("rtdetr-l.pt")
m.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, seed=0,
        optimizer="AdamW", lr0=1e-4, project="runs", name="rtdetr_l_prelim",
        exist_ok=True, plots=True)
r = m.val(data=a.data, split="val", imgsz=640)
print(f"mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} P={r.box.mp:.4f} R={r.box.mr:.4f}")
