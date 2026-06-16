"""Phase 1 first baseline: frozen-backbone linear probe on the PLUS task.

Extracts features with a frozen vision backbone (timm), caches them, then fits
logistic regression and reports:
  in-domain   AUC (train/test within a dataset)
  cross-site  AUC (train on source, test on the OTHER country)

Usage: python probe_plus.py --backbone vit_base_patch14_dinov2.lvd142m [--limit N]
"""
import argparse, os, numpy as np, pandas as pd, torch, timm
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    M = pd.read_csv(f"{ROOT}/data/processed/manifest.csv")
    S = pd.read_csv(f"{ROOT}/data/processed/patient_splits.csv")[["image_path", "split"]]
    M = M.merge(S, on="image_path", how="left")
    return M[M.plus_label.notna()].reset_index(drop=True)   # FARFUM + Czech

@torch.no_grad()
def extract(model, tf, paths, bs=64):
    feats = []
    for i in range(0, len(paths), bs):
        ims = []
        for p in paths[i:i+bs]:
            ims.append(tf(Image.open(p).convert("RGB")))
        x = torch.stack(ims).to(DEV)
        feats.append(model(x).float().cpu().numpy())
        print(f"\r  {min(i+bs,len(paths))}/{len(paths)}", end="", flush=True)
    print()
    return np.concatenate(feats)

def get_feats(df, backbone, model, tf, limit):
    # cache ONLY features keyed by image_path; labels/splits are re-joined at
    # runtime so re-splitting never requires re-extraction.
    cache = ROOT / "data" / "processed" / "feats" / backbone
    os.makedirs(cache, exist_ok=True)
    blobs = {}
    for ds, g in df.groupby("dataset"):
        if limit:
            g = g.groupby("plus_label", group_keys=False).head(limit // 2)
        fp = f"{cache}/{ds}{'_lim'+str(limit) if limit else ''}.npz"
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            paths, X = list(z["paths"]), z["X"]
        else:
            print(f"[extract] {ds} ({len(g)})")
            paths = g.image_path.tolist()
            X = extract(model, tf, paths)
            np.savez(fp, X=X, paths=np.array(paths, dtype=object))
        sub = df.set_index("image_path").loc[paths]
        blobs[ds] = (X, sub.plus_label.to_numpy().astype(int), sub.split.to_numpy())
    return blobs

def fit_eval(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s), average_precision_score(yte, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = load_data()
    print(f"plus-task images: {len(df)} | {DEV}")
    model = timm.create_model(a.backbone, pretrained=True, num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    print("transform:", cfg)
    B = get_feats(df, a.backbone, model, tf, a.limit)

    print("\n=== IN-DOMAIN (train->test, same dataset) ===")
    for ds, (X, y, sp) in B.items():
        tr, te = sp == "train", sp == "test"
        if tr.sum() and te.sum() and len(set(y[tr])) > 1:
            auc, ap = fit_eval(X[tr], y[tr], X[te], y[te])
            print(f"  {ds:8s} train{tr.sum()} test{te.sum()}  AUC {auc:.3f}  AP {ap:.3f}")

    print("\n=== CROSS-SITE (train all source -> test all target) ===")
    names = list(B)
    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            Xs, ys, _ = B[src]; Xt, yt, _ = B[tgt]
            auc, ap = fit_eval(Xs, ys, Xt, yt)
            print(f"  {src:8s} -> {tgt:8s}  AUC {auc:.3f}  AP {ap:.3f}")

if __name__ == "__main__":
    main()
