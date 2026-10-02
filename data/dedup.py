"""Perceptual-hash de-duplication across train/valid/test.
Groups near-identical images (Hamming distance <= --thresh) and moves any
group that straddles splits entirely into train, so val/test stay clean.
Owner: Harish Bharathwaj."""
import argparse, shutil
from pathlib import Path
from PIL import Image
import imagehash

SPLITS = ["train", "valid", "test"]
EXT = {".jpg", ".jpeg", ".png"}

def main(root, thresh, dry):
    root = Path(root)
    recs = []
    for sp in SPLITS:
        for img in sorted((root / sp / "images").glob("*")):
            if img.suffix.lower() not in EXT:
                continue
            try:
                h = imagehash.phash(Image.open(img).convert("RGB"))
            except Exception as e:
                print("skip", img.name, e); continue
            recs.append((sp, img, h))
    print(f"hashed {len(recs)} images")

    parent = list(range(len(recs)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if recs[i][2] - recs[j][2] <= thresh:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(recs)):
        groups.setdefault(find(i), []).append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dup if len({recs[i][0] for i in g}) > 1]
    print(f"duplicate groups: {len(dup)}   spanning splits: {len(cross)}")

    moved = 0
    for g in cross:
        for i in g:
            sp, img, _ = recs[i]
            if sp == "train":
                continue
            lbl = root / sp / "labels" / (img.stem + ".txt")
            print(f"move {sp}/{img.name} -> train")
            if not dry:
                shutil.move(str(img), str(root / "train" / "images" / img.name))
                if lbl.exists():
                    shutil.move(str(lbl), str(root / "train" / "labels" / lbl.name))
            moved += 1
    print(f"moved {moved} files{' (dry run)' if dry else ''}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--thresh", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="report only, move nothing")
    a = ap.parse_args()
    main(a.root, a.thresh, a.dry)
