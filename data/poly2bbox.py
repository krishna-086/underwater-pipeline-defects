"""Convert YOLOv8 polygon (segmentation) labels to bounding-box labels, in place.
Each polygon becomes the axis-aligned box of its min/max extent. Also repairs
lines where two coordinates were written with no space between them.
Run once, before freeze_split.py. Backs up originals to labels_poly/.
Owner: Krishna Anand."""
import argparse, re, shutil
from pathlib import Path

SPLITS = ["train", "valid", "test"]
FUSED = re.compile(r"(?<=\d)(?=0\.)")   # a '0.' glued to the digit before it

def to_box(tokens):
    cls, vals = tokens[0], [float(v) for v in tokens[1:]]
    if len(vals) % 2:
        vals = vals[:-1]
    xs, ys = vals[0::2], vals[1::2]
    if len(xs) < 2:
        return None
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}"

def main(root):
    root = Path(root)
    tot = {"lines": 0, "boxes_kept": 0, "polys": 0, "fused_fixed": 0, "dropped": 0}
    for sp in SPLITS:
        ldir = root / sp / "labels"
        bak = root / sp / "labels_poly"
        if not bak.exists():
            shutil.copytree(ldir, bak)
        for f in sorted(ldir.glob("*.txt")):
            out = []
            for ln in f.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                tot["lines"] += 1
                fixed = FUSED.sub(" ", ln)
                if fixed != ln:
                    tot["fused_fixed"] += 1
                toks = fixed.split()
                if len(toks) == 5:
                    out.append(ln); tot["boxes_kept"] += 1; continue
                box = to_box(toks)
                if box:
                    out.append(box); tot["polys"] += 1
                else:
                    tot["dropped"] += 1
            f.write_text("\n".join(out) + ("\n" if out else ""))
        print(f"{sp}: done")
    print(tot)
    print("originals kept in <split>/labels_poly/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    main(ap.parse_args().root)
