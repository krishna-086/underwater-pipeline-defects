"""RetinaNet ResNet-50 FPN v2. Owner: Siddhartha Saha.
Final run: python train/train_retinanet.py --epochs 100 --name retinanet_100
Shares the loop in torchvision_common.py with Faster R-CNN."""
import argparse
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision_common import run

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="dataset")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--name", default="retinanet_100")
a = ap.parse_args()

m = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
n_anchors = m.head.classification_head.num_anchors
m.head.classification_head = RetinaNetClassificationHead(256, n_anchors, num_classes=3)
run(m, a.name, root=a.root, epochs=a.epochs, batch=a.batch, lr=0.01)
