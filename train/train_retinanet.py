"""RetinaNet ResNet-50 FPN v2. Owner: Siddhartha Saha.
Shares the loop in torchvision_common.py with Faster R-CNN."""
import argparse, math
import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision_common import run

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="dataset")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()

m = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
n_anchors = m.head.classification_head.num_anchors
m.head.classification_head = RetinaNetClassificationHead(256, n_anchors, num_classes=3)  # ids 1,2 used; 0 unused
run(m, "retinanet_prelim", root=a.root, epochs=a.epochs, batch=a.batch, lr=0.01)
