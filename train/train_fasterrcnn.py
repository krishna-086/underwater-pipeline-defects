"""Faster R-CNN ResNet-50 FPN v2. Owner: Pushkar Ojha.
Batch 4 and mixed precision so it fits an 8 GB card at 640."""
import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision_common import run

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="dataset")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()

m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
in_f = m.roi_heads.box_predictor.cls_score.in_features
m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes=3)   # 2 classes + background
run(m, "fasterrcnn_prelim", root=a.root, epochs=a.epochs, batch=a.batch, lr=0.01)
