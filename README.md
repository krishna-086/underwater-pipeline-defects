# Underwater Pipeline Defect Detection: A Four-Architecture Comparison
ICT 4442 Deep Learning mini project, Team 14, MIT Manipal.

Four detectors, one dataset, one protocol.

| Model | Family | Owner | Script |
|---|---|---|---|
| YOLOv8n | one-stage CNN, anchor-free | Krishna Anand | `train/train_yolov8n.py` |
| RT-DETR-L | transformer, end-to-end | Harish Bharathwaj | `train/train_rtdetr.py` |
| Faster R-CNN R50-FPN | two-stage CNN | Pushkar Ojha | `train/train_fasterrcnn.py` |
| RetinaNet R50-FPN | one-stage CNN, anchor-based | Siddhartha Saha | `train/train_retinanet.py` |

## Setup
```
pip install -r requirements.txt
```

## Dataset
Export from Roboflow Universe in **YOLOv8** format and unzip to `dataset/`:
https://universe.roboflow.com/harish-bharathwaj/corossion-detection-6hmzy-fzart

```
dataset/
  data.yaml
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

## Run order
```
python data/dedup.py --root dataset --dry    # 1. report cross-split near-duplicates (drop --dry to move them)
python data/freeze_split.py --root dataset   # 2. integrity check, per-class counts, splits/manifest.json
python data/yolo2coco.py --root dataset      # 3. COCO json for the torchvision models

python train/train_yolov8n.py                # Krishna
python train/train_rtdetr.py                 # Harish
python train/train_fasterrcnn.py             # Pushkar
python train/train_retinanet.py              # Siddhartha

python eval/benchmark_latency.py             # Pushkar, after weights exist
```
Ultralytics metrics: last row of `runs/<name>/results.csv`.
Torchvision metrics: printed by pycocotools at the end of each epoch.

## Layout
- `data/` dataset preparation
- `train/` one script per model, plus the shared torchvision loop
- `eval/` latency benchmark and (later) the common pycocotools comparison
- `splits/` frozen manifest, committed once and never regenerated
- `report/` synopsis, interim and final reports
