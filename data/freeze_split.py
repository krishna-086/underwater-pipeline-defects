"""Label integrity check + frozen split manifest + per-class counts.
Every model trains from splits/manifest.json. Never regenerate it.
Owner: Krishna Anand."""
import argparse, json
from pathlib import Path
from collections import Counter

SPLITS = ["train", "valid", "test"]
EXT = {".jpg", ".jpeg", ".png"}

def read_names(yaml_path):
    names = ["crack", "corrosion"]
    if yaml_path.exists():
        for line in yaml_path.read_text().splitlines():
            if line.strip().startswith("names"):
                raw = line.split(":", 1)[1].strip(" []")
                names = [n.strip(" '\"") for n in raw.split(",") if n.strip()]
    return names

def main(root, out):
    root = Path(root)
    names = read_names(root / "data.yaml")
    manifest, bad = {"classes": names}, []
    for sp in SPLITS:
        imgs = sorted(p.name for p in (root / sp / "images").glob("*") if p.suffix.lower() in EXT)
        cnt = Counter()
        for name in imgs:
            lbl = root / sp / "labels" / (Path(name).stem + ".txt")
            if not lbl.exists():
                bad.append(f"{sp}/{name}: missing label"); continue
            for ln in lbl.read_text().splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    bad.append(f"{sp}/{lbl.name}: bad line '{ln}'"); continue
                c, *xywh = parts
                c = int(c)
                if not 0 <= c < len(names):
                    bad.append(f"{sp}/{lbl.name}: class {c} out of range")
                if any(not 0.0 <= float(v) <= 1.0 for v in xywh):
                    bad.append(f"{sp}/{lbl.name}: coord outside [0,1]")
                cnt[names[c] if 0 <= c < len(names) else str(c)] += 1
        manifest[sp] = {"n_images": len(imgs), "instances": dict(cnt), "images": imgs}
        print(f"{sp:6s} images={len(imgs):5d}  " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    print(f"integrity problems: {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="splits/manifest.json")
    a = ap.parse_args()
    main(a.root, a.out)
