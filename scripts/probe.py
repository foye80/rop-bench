"""ROP-Bench frozen-backbone linear probe — supports both tasks.

Features are extracted ONCE per dataset (all images), cached by image_path, and
reused across tasks and backbones. Labels/splits are joined at runtime.

Reports in-domain AUC/AP (train->test same dataset) and cross-site AUC/AP
(train on each source dataset, test on every other dataset).

Usage:
  python probe.py --task presence --backbone vit_base_patch14_dinov2.lvd142m
  python probe.py --task plus     --backbone <hf-or-timm-name>
"""
import argparse, os, numpy as np, pandas as pd, torch, timm
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASKCOL = {"plus": "plus_label", "presence": "rop_presence"}

def build_backbone(name, img_size=0):
    """Returns (model, transform). Special-cases RETFound (ViT-L MAE encoder).
    img_size>0 overrides the backbone's default input resolution (for fair
    cross-backbone comparison)."""
    if name == "retfound_cfp":
        m = timm.create_model("vit_large_patch16_224", pretrained=False,
                              num_classes=0, global_pool="token")   # CLS + encoder norm
        sd = torch.load(ROOT / "pretrained" / "RETFound_cfp_weights.pth",
                        map_location="cpu", weights_only=False)["model"]
        msg = m.load_state_dict(sd, strict=False)
        assert not [k for k in msg.missing_keys if "head" not in k], msg.missing_keys
        print("[RETFound] loaded; ignored MAE-decoder keys:",
              len([k for k in msg.unexpected_keys]))
    else:
        kw = dict(img_size=img_size) if img_size else {}
        m = timm.create_model(name, pretrained=True, num_classes=0, **kw)
    m = m.eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=m)
    if img_size:
        cfg["input_size"] = (3, img_size, img_size)
    print("transform:", cfg)
    return m, timm.data.create_transform(**cfg)

def load_manifest():
    M = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    S = pd.read_csv(ROOT / "data" / "processed" / "patient_splits.csv")[["image_path", "split"]]
    return M.merge(S, on="image_path", how="left")

@torch.no_grad()
def extract(model, tf, paths, bs=48):
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]).to(DEV)
        feats.append(model(x).float().cpu().numpy())
        print(f"\r  {min(i+bs,len(paths))}/{len(paths)}", end="", flush=True)
    print()
    return np.concatenate(feats)

def dataset_feats(ds, paths_all, cache_tag, model, tf):
    """Extract/cache ALL images of a dataset -> dict path->vector."""
    cache = ROOT / "data" / "processed" / "feats" / cache_tag
    os.makedirs(cache, exist_ok=True)
    fp = cache / f"{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=True)
        cached = dict(zip(z["paths"], z["X"]))
        missing = [p for p in paths_all if p not in cached]
        if not missing:
            return cached
        print(f"[extract+] {ds}: {len(missing)} new")
        Xn = extract(model, tf, missing)
        cached.update(dict(zip(missing, Xn)))
    else:
        print(f"[extract] {ds} ({len(paths_all)})")
        cached = dict(zip(paths_all, extract(model, tf, paths_all)))
    np.savez(fp, X=np.stack([cached[p] for p in paths_all]),
             paths=np.array(paths_all, dtype=object))
    return cached

def fit_eval(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s), average_precision_score(yte, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKCOL), required=True)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img_size", type=int, default=0, help="override input res")
    a = ap.parse_args()
    cache_tag = a.backbone + (f"_{a.img_size}" if a.img_size else "")
    col = TASKCOL[a.task]
    M = load_manifest()
    task = M[M[col].notna()].copy()
    print(f"[{a.task}] images={len(task)}  datasets={sorted(task.dataset.unique())}  {DEV}")

    model, tf = build_backbone(a.backbone, a.img_size)

    blobs = {}
    for ds, g in task.groupby("dataset"):
        all_paths = M[M.dataset == ds].image_path.tolist()      # extract whole dataset once
        feat = dataset_feats(ds, all_paths, cache_tag, model, tf)
        X = np.stack([feat[p] for p in g.image_path])
        blobs[ds] = (X, g[col].to_numpy().astype(int), g.split.to_numpy())

    print("\n=== IN-DOMAIN (train->test) ===")
    for ds, (X, y, sp) in blobs.items():
        tr, te = sp == "train", sp == "test"
        if tr.sum() and te.sum() and len(set(y[tr])) > 1 and len(set(y[te])) > 1:
            auc, apr = fit_eval(X[tr], y[tr], X[te], y[te])
            print(f"  {ds:9s} train{tr.sum():5d}({y[tr].sum():4d}+) test{te.sum():5d}"
                  f"({y[te].sum():4d}+)  AUC {auc:.3f}  AP {apr:.3f}")
        else:
            print(f"  {ds:9s} skipped (single-class split)")

    print("\n=== CROSS-SITE (train all source -> test all target) ===")
    for src in blobs:
        for tgt in blobs:
            if src != tgt and len(set(blobs[src][1])) > 1 and len(set(blobs[tgt][1])) > 1:
                auc, apr = fit_eval(blobs[src][0], blobs[src][1], blobs[tgt][0], blobs[tgt][1])
                print(f"  {src:9s} -> {tgt:9s}  AUC {auc:.3f}  AP {apr:.3f}")

if __name__ == "__main__":
    main()
