"""Phase 2 — fine-tuning label-efficiency: RETFound vs DINOv2 vs random init.

Trains on Czech train split (presence task) at a label fraction, evaluates
in-domain (Czech test) and cross-site (Shenzhen, HVDROPDB). All @224, class-
weighted CE. Fraction subsampling is patient-stratified by any-positive.

Usage: python finetune.py --init {retfound,dinov2,random} --frac 0.1 [--epochs 15]
Appends one result line to data/processed/finetune_results.csv.
"""
import argparse, os, numpy as np, pandas as pd, torch, timm
from pathlib import Path
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
DEV = "cuda"
IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def build(init):
    if init == "retfound":
        m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=2,
                              global_pool="token")
        sd = torch.load(ROOT / "pretrained" / "RETFound_cfp_weights.pth",
                        map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd, strict=False)
    elif init == "dinov2":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=2, img_size=224)
    elif init == "random":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False,
                              num_classes=2, img_size=224)
    return m.to(DEV)

class DS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        import torchvision.transforms as T
        if train:
            self.tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                                 T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize(**IMAGENET)])
        else:
            self.tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                                 T.Normalize(**IMAGENET)])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r.image_path).convert("RGB")), int(r.rop_presence)

def loader(df, train, bs=32):
    return DataLoader(DS(df, train), batch_size=bs, shuffle=train,
                      num_workers=3, pin_memory=True, drop_last=train)

@torch.no_grad()
def auc(model, df):
    model.eval(); ys, ss = [], []
    for x, y in loader(df, False, 64):
        p = torch.softmax(model(x.to(DEV)), 1)[:, 1].cpu().numpy()
        ss.append(p); ys.append(y.numpy())
    y, s = np.concatenate(ys), np.concatenate(ss)
    return roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")

def subsample(df, frac, seed=0):
    if frac >= 1.0: return df
    rng = np.random.default_rng(seed)
    pos = df.groupby("patient_id").rop_presence.max()
    keep = []
    for lab in (0, 1):
        ps = pos.index[pos.values == lab].to_numpy()
        keep += list(rng.choice(ps, max(1, int(round(len(ps) * frac))), replace=False))
    return df[df.patient_id.isin(keep)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["retfound", "dinov2", "random"], required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    M = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv").merge(
        pd.read_csv(ROOT / "data" / "processed" / "patient_splits.csv")[["image_path", "split"]],
        on="image_path")
    P = M[M.rop_presence.notna()]
    tr = subsample(P[(P.dataset == "Czech") & (P.split == "train")], a.frac)
    va = P[(P.dataset == "Czech") & (P.split == "val")]
    tests = {"czech_in": P[(P.dataset == "Czech") & (P.split == "test")],
             "shenzhen": P[P.dataset == "Shenzhen"], "hvdropdb": P[P.dataset == "HVDROPDB"]}
    print(f"init={a.init} frac={a.frac} train={len(tr)}({int(tr.rop_presence.sum())}+) "
          f"patients={tr.patient_id.nunique()}")

    model = build(a.init)
    w = torch.tensor([1.0, max(1.0, (tr.rop_presence == 0).sum() / max(1, (tr.rop_presence == 1).sum()))],
                     device=DEV).float()
    crit = nn.CrossEntropyLoss(weight=w)
    head = [p for n, p in model.named_parameters() if "head" in n]
    body = [p for n, p in model.named_parameters() if "head" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-5}, {"params": head, "lr": 1e-3}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = loader(tr, True)
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        for x, y in dl:
            opt.zero_grad()
            loss = crit(model(x.to(DEV)), y.to(DEV))
            loss.backward(); opt.step()
        sched.step()
        v = auc(model, va) if len(va) and va.rop_presence.nunique() > 1 else float("nan")
        if not np.isnan(v) and v > best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
        print(f"  ep{ep} loss{loss.item():.3f} val_auc{v:.3f}", flush=True)
    if best_state: model.load_state_dict(best_state)

    res = {"init": a.init, "frac": a.frac, "n_train": len(tr),
           "val_auc": round(best, 4)}
    for name, df in tests.items():
        res[name] = round(auc(model, df), 4)
    print("RESULT", res)
    fp = ROOT / "data" / "processed" / "finetune_results.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)

if __name__ == "__main__":
    main()
