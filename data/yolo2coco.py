"""YOLO txt labels -> COCO json per split, for torchvision training and
pycocotools scoring. Category ids start at 1 (0 is background in torchvision).
Owner: Harish Bharathwaj."""
import argparse, json
from pathlib import Path
from PIL import Image

EXT = {".jpg", ".jpeg", ".png"}

def convert(root, sp, names):
    root = Path(root)
    images, anns, aid = [], [], 1
    for iid, img in enumerate(sorted(p for p in (root / sp / "images").glob("*") if p.suffix.lower() in EXT), 1):
        W, H = Image.open(img).size
        images.append({"id": iid, "file_name": img.name, "width": W, "height": H})
        lbl = root / sp / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        for ln in lbl.read_text().splitlines():
            c, cx, cy, w, h = ln.split()
            w, h = float(w) * W, float(h) * H
            x, y = float(cx) * W - w / 2, float(cy) * H - h / 2
            anns.append({"id": aid, "image_id": iid, "category_id": int(c) + 1,
                         "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
            aid += 1
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    out = root / sp / "annotations.json"
    out.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    print(f"{sp}: {len(images)} images, {len(anns)} boxes -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--names", nargs="+", default=["crack", "corrosion"])
    a = ap.parse_args()
    for sp in ["train", "valid", "test"]:
        convert(a.root, sp, a.names)
