"""Export every miss, false positive and class confusion of one model on the
validation split as annotated images, for hand-sorting into the Table V crack
failure taxonomy. Ground truth drawn green, predictions red. Owner: Harish Bharathwaj.

Usage:
  python eval/error_analysis.py --kind yolo --weights runs/yolov8n_100/weights/best.pt
  kinds: yolo | rtdetr | frcnn | retina
"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0

def predict(kind, weights, img_path, conf):
    if kind in ("yolo", "rtdetr"):
        from ultralytics import YOLO, RTDETR
        if not hasattr(predict, "m"):
            predict.m = (RTDETR if kind == "rtdetr" else YOLO)(weights)
        r = predict.m.predict(str(img_path), imgsz=640, conf=conf, verbose=False)[0]
        return [(int(c) + 1, float(s), [b[0], b[1], b[2] - b[0], b[3] - b[1]])
                for b, s, c in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist())]
    import torch
    import torchvision.transforms.functional as TF
    from export_predictions import build_tv
    if not hasattr(predict, "m"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = build_tv(kind); m.load_state_dict(torch.load(weights, map_location=dev))
        predict.m, predict.dev = m.eval().to(dev), dev
    x = TF.normalize(TF.to_tensor(Image.open(img_path).convert("RGB")),
                     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(predict.dev)
    with __import__("torch").no_grad():
        o = predict.m([x])[0]
    return [(int(c), float(s), [b[0], b[1], b[2] - b[0], b[3] - b[1]])
            for b, s, c in zip(o["boxes"].cpu().tolist(), o["scores"].cpu().tolist(), o["labels"].cpu().tolist())
            if s >= 0.25]

def draw(img, gts, preds, note):
    im = img.convert("RGB"); d = ImageDraw.Draw(im)
    for cid, b in gts:
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(0, 200, 0), width=3)
    for cid, s, b in preds:
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(220, 0, 0), width=2)
        d.text((b[0] + 2, b[1] + 2), f"{s:.2f}", fill=(220, 0, 0))
    d.text((4, 4), note, fill=(255, 255, 0))
    return im

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["yolo", "rtdetr", "frcnn", "retina"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    gt = json.loads((Path(a.root) / a.split / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in gt["categories"]}
    anns = {}
    for an in gt["annotations"]:
        anns.setdefault(an["image_id"], []).append((an["category_id"], an["bbox"]))
    out = Path("eval/error_analysis")
    for sub in ("misses", "false_positives", "class_confusions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    counts = {"miss": {c: 0 for c in cat}, "fp": {c: 0 for c in cat}, "confusion": 0}
    for im in gt["images"]:
        ip = Path(a.root) / a.split / "images" / im["file_name"]
        preds = predict(a.kind, a.weights, ip, a.conf)
        gts = anns.get(im["id"], [])
        g_used = [False] * len(gts); p_used = [False] * len(preds)
        for pi, (pc, s, pb) in sorted(enumerate(preds), key=lambda t: -t[1][1]):
            best, bi = 0.0, -1
            for gi, (gc, gb) in enumerate(gts):
                if g_used[gi] or gc != pc:
                    continue
                v = iou(pb, gb)
                if v > best:
                    best, bi = v, gi
            if best >= a.thr:
                g_used[bi] = p_used[pi] = True
        img = None
        for gi, (gc, gb) in enumerate(gts):
            if g_used[gi]:
                continue
            conf_hit = any(not p_used[pi] and iou(pb, gb) >= a.thr
                           for pi, (pc, s, pb) in enumerate(preds) if pc != gc)
            img = img or Image.open(ip)
            kind = "class_confusions" if conf_hit else "misses"
            tag = f"missed {cat[gc]}" + (" (predicted as other class)" if conf_hit else "")
            draw(img, [(gc, gb)], [p for pi, p in enumerate(preds) if not p_used[pi]], tag) \
                .save(out / kind / f"{Path(im['file_name']).stem}_{cat[gc]}.jpg")
            if conf_hit:
                counts["confusion"] += 1
            else:
                counts["miss"][gc] += 1
        for pi, (pc, s, pb) in enumerate(preds):
            if p_used[pi]:
                continue
            img = img or Image.open(ip)
            draw(img, gts, [(pc, s, pb)], f"false positive {cat[pc]}") \
                .save(out / "false_positives" / f"{Path(im['file_name']).stem}_{cat[pc]}_{s:.2f}.jpg")
            counts["fp"][pc] += 1
    print("missed, per class:", {cat[c]: n for c, n in counts["miss"].items()})
    print("class confusions:", counts["confusion"])
    print("false positives, per class:", {cat[c]: n for c, n in counts["fp"].items()})
    print(f"images written under {out}/ -- sort them by hand into the Table V categories and fill the counts")
