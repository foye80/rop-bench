"""Cluster-bootstrap CIs for the DINOv2@224 vs RETFound@224 comparison (presence task).

Resamples TEST patients (clusters) with replacement; HVDROPDB has no patient_id so
each image is its own cluster. Reports per-backbone AUC [2.5,97.5] and the PAIRED
difference (DINOv2 - RETFound) CI on identical resamples — gap is significant iff the
difference CI excludes 0. LR is trained once per direction (features are cached).
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
B, SEED = 2000, 0
rng = np.random.default_rng(SEED)
TAGS = {"DINOv2@224": "vit_base_patch14_dinov2.lvd142m_224", "RETFound@224": "retfound_cfp"}

M = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv").merge(
    pd.read_csv(ROOT / "data" / "processed" / "patient_splits.csv")[["image_path", "split"]],
    on="image_path")
task = M[M.rop_presence.notna()].copy()
task["grp"] = task.patient_id.fillna(task.image_path)        # HVDROPDB -> image-level
DATASETS = sorted(task.dataset.unique())

def feats(tag):
    d = {}
    for ds in DATASETS:
        z = np.load(ROOT / "data" / "processed" / "feats" / tag / f"{ds}.npz", allow_pickle=True)
        d.update(dict(zip(z["paths"], z["X"])))
    return d
F = {name: feats(tag) for name, tag in TAGS.items()}

def Xof(name, df):
    return np.stack([F[name][p] for p in df.image_path])

def train_idx(src, tgt):
    g = task[task.dataset == src]
    if src == tgt:                                   # in-domain: train split only
        return g[g.split == "train"]
    return g                                          # cross-site: ALL source (matches probe.py)

def test_idx(src, tgt):
    if src == tgt:                                   # in-domain
        return task[(task.dataset == src) & (task.split == "test")]
    return task[task.dataset == tgt]                 # cross-site: whole target

def scores(name, tr, te):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xof(name, tr), tr.rop_presence.astype(int))
    return clf.predict_proba(Xof(name, te))[:, 1]

def boot(y, sd, sr, grp):
    uniq = np.unique(grp)
    idxby = {g: np.where(grp == g)[0] for g in uniq}
    ad, ar, diff = [], [], []
    for _ in range(B):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idxby[g] for g in samp])
        if len(set(y[idx])) < 2:
            continue
        a, b = roc_auc_score(y[idx], sd[idx]), roc_auc_score(y[idx], sr[idx])
        ad.append(a); ar.append(b); diff.append(a - b)
    pc = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return (np.mean(ad), pc(ad)), (np.mean(ar), pc(ar)), (np.mean(diff), pc(diff))

pairs = [(s, s) for s in ("Czech", "Shenzhen")] + \
        [(s, t) for s in DATASETS for t in DATASETS if s != t]
print(f"{'direction':22s} {'DINOv2@224':>20s} {'RETFound@224':>20s} "
      f"{'diff(D-R)':>22s}  sig")
for src, tgt in pairs:
    tr, te = train_idx(src, tgt), test_idx(src, tgt)
    y = te.rop_presence.to_numpy().astype(int)
    if len(set(y)) < 2 or len(tr) == 0:
        continue
    sd = scores("DINOv2@224", tr, te); sr = scores("RETFound@224", tr, te)
    (md, cd), (mr, cr), (mdiff, cdiff) = boot(y, sd, sr, te.grp.to_numpy())
    sig = "***" if (cdiff[0] > 0 or cdiff[1] < 0) else "ns"
    name = f"{src[:4]}->{tgt[:4]}" + ("(in)" if src == tgt else "")
    print(f"{name:22s} {md:.3f}[{cd[0]:.3f},{cd[1]:.3f}] "
          f"{mr:.3f}[{cr[0]:.3f},{cr[1]:.3f}] "
          f"{mdiff:+.3f}[{cdiff[0]:+.3f},{cdiff[1]:+.3f}] {sig}")
