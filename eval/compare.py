"""The single scoring harness. Reads the ground truth and every prediction JSON
from eval/export_predictions.py, scores all models with pycocotools, bootstraps
a 95% CI, and prints Table III and Table IV of the final report verbatim.
Owner: Krishna Anand."""
import argparse, contextlib, copy, io, json, random
from pathlib import Path

def quiet_coco(gt_dict):
    from pycocotools.coco import COCO
    c = COCO()
    c.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c

def coco_map(gt_dict, dets, cat_ids=None):
    from pycocotools.cocoeval import COCOeval
    if not dets:
        return 0.0, 0.0
    gt = quiet_coco(gt_dict)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(dets))
        e = COCOeval(gt, dt, "bbox")
        if cat_ids:
            e.params.catIds = cat_ids
        e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[1], e.stats[0]   # mAP@0.5, mAP@0.5:0.95

def resample(gt_dict, dets, rng):
    """Bootstrap draw: sample images with replacement, remap duplicated ids."""
    ids = [im["id"] for im in gt_dict["images"]]
    pick = [rng.choice(ids) for _ in ids]
    im_by_id = {im["id"]: im for im in gt_dict["images"]}
    anns_by_img, dets_by_img = {}, {}
    for a in gt_dict["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    for d in dets:
        dets_by_img.setdefault(d["image_id"], []).append(d)
    n_img, n_ann = [], []
    n_det, aid = [], 1
    for new_id, old in enumerate(pick, 1):
        im = dict(im_by_id[old]); im["id"] = new_id; n_img.append(im)
        for a in anns_by_img.get(old, []):
            a = dict(a); a["id"] = aid; a["image_id"] = new_id; aid += 1; n_ann.append(a)
        for d in dets_by_img.get(old, []):
            d = dict(d); d["image_id"] = new_id; n_det.append(d)
    return {"images": n_img, "annotations": n_ann, "categories": gt_dict["categories"]}, n_det

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0

def prf(gt_dict, dets, conf=0.25, thr=0.5):
    """Greedy IoU-0.5 matching at a fixed confidence. Returns micro P/R/F1 and per-class recall."""
    anns_by = {}
    for a in gt_dict["annotations"]:
        anns_by.setdefault((a["image_id"], a["category_id"]), []).append(a["bbox"])
    dets_by = {}
    for d in dets:
        if d["score"] >= conf:
            dets_by.setdefault((d["image_id"], d["category_id"]), []).append((d["score"], d["bbox"]))
    cats = [c["id"] for c in gt_dict["categories"]]
    tp = {c: 0 for c in cats}; fp = {c: 0 for c in cats}; ngt = {c: 0 for c in cats}
    img_ids = [im["id"] for im in gt_dict["images"]]
    for img in img_ids:
        for c in cats:
            gts = anns_by.get((img, c), []); ngt[c] += len(gts)
            used = [False] * len(gts)
            for s, box in sorted(dets_by.get((img, c), []), reverse=True):
                best, bi = 0.0, -1
                for i, g in enumerate(gts):
                    if used[i]:
                        continue
                    v = iou(box, g)
                    if v > best:
                        best, bi = v, i
                if best >= thr:
                    used[bi] = True; tp[c] += 1
                else:
                    fp[c] += 1
    TP, FP, NG = sum(tp.values()), sum(fp.values()), sum(ngt.values())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / NG if NG else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    per_rec = {c: (tp[c] / ngt[c] if ngt[c] else 0.0) for c in cats}
    return P, R, F1, per_rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="dataset/test/annotations.json")
    ap.add_argument("--preds", default="eval/preds")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws (0 to skip; 1000 takes a few min/model)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text())
    cat_name = {c["id"]: c["name"] for c in gt["categories"]}
    rows, per_class = [], []
    for pj in sorted(Path(a.preds).glob("*.json")):
        name = pj.stem
        dets = json.loads(pj.read_text())
        m50, m5095 = coco_map(gt, dets)
        P, R, F1, rec = prf(gt, dets)
        lo = hi = float("nan")
        if a.boot:
            rng = random.Random(a.seed)
            vals = []
            for _ in range(a.boot):
                g2, d2 = resample(gt, dets, rng)
                vals.append(coco_map(g2, d2)[1])
            vals.sort()
            lo, hi = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]
        rows.append((name, m50, m5095, lo, hi, P, R, F1))
        pc = [name]
        for cid in sorted(cat_name):
            ap50, _ = coco_map(gt, dets, cat_ids=[cid])
            pc += [ap50, rec[cid]]
        per_class.append(pc)
        print(f"done {name}")
    print("\n=== TABLE III (test split, pycocotools) ===")
    print(f"{'model':14s} {'mAP@0.5':>8s} {'mAP@.5:.95':>10s} {'95% CI':>17s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for n, m50, m, lo, hi, P, R, F1 in rows:
        print(f"{n:14s} {m50:8.3f} {m:10.3f}   [{lo:.3f}, {hi:.3f}] {P:6.3f} {R:6.3f} {F1:6.3f}")
    names = [cat_name[c] for c in sorted(cat_name)]
    print("\n=== TABLE IV (per class) ===")
    print(f"{'model':14s} " + " ".join(f"{nm+' AP50':>12s} {nm+' rec':>10s}" for nm in names))
    for pc in per_class:
        print(f"{pc[0]:14s} " + " ".join(f"{pc[1+2*i]:12.3f} {pc[2+2*i]:10.3f}" for i in range(len(names))))
