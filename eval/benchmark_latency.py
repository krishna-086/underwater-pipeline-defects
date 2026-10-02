"""Inference latency and FPS for all four models on one machine, same input,
batch 1, 640x640, fp32, 50 warm-up + 200 timed iterations. Owner: Pushkar Ojha.
Run after the weights exist; models whose weights are missing are skipped."""
import time
from pathlib import Path
import torch

def bench(fn, x, warm=50, iters=200):
    for _ in range(warm):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    return ms, 1000 / ms

def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(1, 3, 640, 640, device=dev)
rows = []

for name, w in [("YOLOv8n", "runs/yolov8n_prelim/weights/best.pt"),
                ("RT-DETR-L", "runs/rtdetr_l_prelim/weights/best.pt")]:
    if Path(w).exists():
        from ultralytics import YOLO, RTDETR
        m = (RTDETR if "rtdetr" in w else YOLO)(w)
        ms, fps = bench(lambda t: m.predict(t, imgsz=640, verbose=False, device=dev.type), x)
        rows.append((name, count(m.model), ms, fps))

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
tv = []
if Path("runs/fasterrcnn_prelim/best.pt").exists():
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 3)
    m.load_state_dict(torch.load("runs/fasterrcnn_prelim/best.pt", map_location=dev)); tv.append(("Faster R-CNN R50-FPN", m))
if Path("runs/retinanet_prelim/best.pt").exists():
    m = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=None)
    m.head.classification_head = RetinaNetClassificationHead(256, m.head.classification_head.num_anchors, 3)
    m.load_state_dict(torch.load("runs/retinanet_prelim/best.pt", map_location=dev)); tv.append(("RetinaNet R50-FPN", m))
for name, m in tv:
    m.eval().to(dev)
    with torch.no_grad():
        ms, fps = bench(lambda t: m([t[0]]), x)
    rows.append((name, count(m), ms, fps))

print(f"{'model':22s} {'params(M)':>10s} {'ms/img':>8s} {'FPS':>7s}   device={dev}")
for n, pm, ms, fps in rows:
    print(f"{n:22s} {pm:10.1f} {ms:8.1f} {fps:7.1f}")
