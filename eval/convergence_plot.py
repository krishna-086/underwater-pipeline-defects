"""Validation mAP@0.5 against epoch for all four runs, one axis, with the epoch
at which each model reaches 95% of its own final value marked. Reads Ultralytics
results.csv and the torchvision loop's results.json. Owner: Siddhartha Saha.

Usage:
  python eval/convergence_plot.py yolov8n=runs/yolov8n_100/results.csv \
      rtdetr=runs/rtdetr_l_100/results.csv \
      fasterrcnn=runs/fasterrcnn_100/results.json \
      retinanet=runs/retinanet_100/results.json
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_curve(path):
    p = Path(path)
    if p.suffix == ".csv":
        import csv
        with open(p) as f:
            rows = list(csv.DictReader(f))
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        return [float(r["metrics/mAP50(B)"]) for r in rows]
    log = json.loads(p.read_text())
    return [e["mAP50"] for e in log]

def main(pairs):
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=200)
    styles = ["-", "--", "-.", ":"]
    for (name, path), ls in zip(pairs, styles):
        y = load_curve(path)
        x = list(range(1, len(y) + 1))
        ax.plot(x, y, ls, lw=1.4, label=name)
        final = y[-1]
        reach = next((i + 1 for i, v in enumerate(y) if v >= 0.95 * final), len(y))
        ax.plot([reach], [y[reach - 1]], "o", ms=4, color=ax.lines[-1].get_color())
        print(f"{name:12s} final mAP@0.5 = {final:.3f}   95%-of-final at epoch {reach}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mAP@0.5")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = "eval/fig_convergence.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

if __name__ == "__main__":
    pairs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    if not pairs:
        sys.exit(__doc__)
    main(pairs)
