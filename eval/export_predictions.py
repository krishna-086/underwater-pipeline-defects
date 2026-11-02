"""Run a trained model on a split and export COCO-format detections to eval/preds/.
One JSON per model. These JSONs are the single input to eval/compare.py.
Owner: Krishna Anand."""
import argparse, json
from pathlib import Path

def load_gt(root, split):
    gt = json.loads((Path(root) / split / "annotations.json").read_text())
    return gt, {im["file_name"]: im["id"] for im in gt["images"]}

def export_ultra(weights, kind, root, split, name2id):
    from ultralytics import YOLO, RTDETR
    m = (RTDETR if kind == "rtdetr" else YOLO)(weights)
    dets = []
    for img in sorted((Path(root) / split / "images").glob("*")):
        if img.name not in name2id:
            continue
        r = m.predict(str(img), imgsz=640, conf=0.001, verbose=False)[0]
        for b, s, c in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
            x1, y1, x2, y2 = b
            dets.append({"image_id": name2id[img.name], "category_id": int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return dets

def build_tv(kind):
    import torchvision
    if kind == "frcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 3)
    else:
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        m = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=None)
        m.head.classification_head = RetinaNetClassificationHead(256, m.head.classification_head.num_anchors, 3)
    return m

def export_tv(weights, kind, root, split, name2id):
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_tv(kind)
    m.load_state_dict(torch.load(weights, map_location=dev))
    m.eval().to(dev)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    dets = []
    with torch.no_grad():
        for img in sorted((Path(root) / split / "images").glob("*")):
            if img.name not in name2id:
                continue
            x = TF.normalize(TF.to_tensor(Image.open(img).convert("RGB")), MEAN, STD).to(dev)
            o = m([x])[0]
            for b, s, c in zip(o["boxes"].cpu().tolist(), o["scores"].cpu().tolist(), o["labels"].cpu().tolist()):
                x1, y1, x2, y2 = b
                dets.append({"image_id": name2id[img.name], "category_id": int(c),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return dets

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--yolo", help="path to YOLOv8n best.pt")
    ap.add_argument("--rtdetr", help="path to RT-DETR-L best.pt")
    ap.add_argument("--frcnn", help="path to Faster R-CNN best.pt")
    ap.add_argument("--retina", help="path to RetinaNet best.pt")
    a = ap.parse_args()
    _, name2id = load_gt(a.root, a.split)
    out = Path("eval/preds"); out.mkdir(parents=True, exist_ok=True)
    jobs = [("yolov8n", a.yolo, "ultra", "yolo"), ("rtdetr_l", a.rtdetr, "ultra", "rtdetr"),
            ("fasterrcnn", a.frcnn, "tv", "frcnn"), ("retinanet", a.retina, "tv", "retina")]
    for name, w, fam, kind in jobs:
        if not w:
            print(f"{name}: no weights given, skipped"); continue
        if not Path(w).exists():
            print(f"{name}: {w} not found, skipped"); continue
        dets = export_ultra(w, kind, a.root, a.split, name2id) if fam == "ultra" \
               else export_tv(w, kind, a.root, a.split, name2id)
        (out / f"{name}_{a.split}.json").write_text(json.dumps(dets))
        print(f"{name}: {len(dets)} detections -> {out}/{name}_{a.split}.json")
