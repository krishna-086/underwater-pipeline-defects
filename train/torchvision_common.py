"""Shared pieces for the two torchvision detectors: Dataset over the COCO json
written by data/yolo2coco.py, collate, one training epoch, pycocotools eval.
Owner: Siddhartha Saha. Used by train_fasterrcnn.py and train_retinanet.py."""
import json, random, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class CocoDet(Dataset):
    def __init__(self, root, split, train=False):
        self.dir = Path(root) / split / "images"
        self.coco = COCO(str(Path(root) / split / "annotations.json"))
        self.ids = sorted(self.coco.imgs.keys())
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        info = self.coco.imgs[iid]
        img = Image.open(self.dir / info["file_name"]).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid))
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                              for a in anns], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)
        if self.train and random.random() < 0.5:
            img = F.hflip(img)
            w = info["width"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        x = F.normalize(F.to_tensor(img), MEAN, STD)
        return x, {"boxes": boxes, "labels": labels, "image_id": iid}

def collate(batch):
    return tuple(zip(*batch))

def loaders(root, batch, workers=4):
    tr = DataLoader(CocoDet(root, "train", train=True), batch_size=batch, shuffle=True,
                    num_workers=workers, collate_fn=collate)
    va = DataLoader(CocoDet(root, "valid"), batch_size=batch, shuffle=False,
                    num_workers=workers, collate_fn=collate)
    return tr, va

def train_one_epoch(model, opt, loader, dev, scaler):
    model.train()
    tot, n, t0 = 0.0, 0, time.time()
    for imgs, tgts in loader:
        imgs = [i.to(dev) for i in imgs]
        tgts = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
        with torch.autocast(device_type=dev.type, enabled=scaler is not None):
            loss = sum(model(imgs, tgts).values())
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        tot += loss.item(); n += 1
    return tot / max(n, 1), time.time() - t0

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    res = []
    for imgs, tgts in loader:
        out = model([i.to(dev) for i in imgs])
        for o, t in zip(out, tgts):
            for b, s, l in zip(o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu()):
                x1, y1, x2, y2 = b.tolist()
                res.append({"image_id": t["image_id"], "category_id": int(l),
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    gt = loader.dataset.coco
    if not res:
        print("no detections"); return 0.0, 0.0
    dt = gt.loadRes(res)
    ev = COCOeval(gt, dt, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[1], ev.stats[0]   # mAP@0.5, mAP@0.5:0.95

def run(model, name, root="dataset", epochs=20, batch=4, lr=0.01):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    tr, va = loaders(root, batch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.7)], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler() if dev.type == "cuda" else None
    out = Path("runs") / name; out.mkdir(parents=True, exist_ok=True)
    best, log = 0.0, []
    for ep in range(1, epochs + 1):
        loss, dt = train_one_epoch(model, opt, tr, dev, scaler)
        sched.step()
        m50, m5095 = evaluate(model, va, dev)
        log.append({"epoch": ep, "loss": loss, "mAP50": m50, "mAP50-95": m5095, "sec": dt})
        print(f"epoch {ep:3d} loss {loss:.4f} mAP50 {m50:.4f} mAP50-95 {m5095:.4f} ({dt:.0f}s)")
        if m50 > best:
            best = m50; torch.save(model.state_dict(), out / "best.pt")
        (out / "results.json").write_text(json.dumps(log, indent=1))
    print(f"best mAP50 {best:.4f} -> {out}")
