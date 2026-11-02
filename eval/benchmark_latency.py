"""Inference latency and FPS for all four models on one machine, same input,
batch 1, 640x640, FP32, 50 warm-up + 200 timed iterations. Owner: Pushkar Ojha.
Weight paths are arguments; models with no weights given are skipped.

Usage:
  python eval/benchmark_latency.py --yolo runs/yolov8n_100/weights/best.pt \
      --rtdetr runs/rtdetr_l_100/weights/best.pt \
      --frcnn runs/fasterrcnn_100/best.pt --retina runs/retinanet_100/best.pt \
      | tee eval/latency_results.txt
"""
import argparse, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_predictions import build_tv

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo"); ap.add_argument("--rtdetr")
    ap.add_argument("--frcnn"); ap.add_argument("--retina")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 640, 640, device=dev)
    rows = []
    for name, w, kind in [("YOLOv8n", a.yolo, "yolo"), ("RT-DETR-L", a.rtdetr, "rtdetr")]:
        if w and Path(w).exists():
            from ultralytics import YOLO, RTDETR
            m = (RTDETR if kind == "rtdetr" else YOLO)(w)
            ms, fps = bench(lambda t: m.predict(t, imgsz=640, verbose=False, device=dev.type), x)
            rows.append((name, count(m.model), ms, fps))
    for name, w, kind in [("Faster R-CNN R50-FPN", a.frcnn, "frcnn"), ("RetinaNet R50-FPN", a.retina, "retina")]:
        if w and Path(w).exists():
            m = build_tv(kind)
            m.load_state_dict(torch.load(w, map_location=dev))
            m.eval().to(dev)
            with torch.no_grad():
                ms, fps = bench(lambda t: m([t[0]]), x)
            rows.append((name, count(m), ms, fps))
    print(f"{'model':22s} {'params(M)':>10s} {'ms/img':>8s} {'FPS':>7s}   device={dev}")
    for n, pm, ms, fps in rows:
        print(f"{n:22s} {pm:10.1f} {ms:8.1f} {fps:7.1f}")
