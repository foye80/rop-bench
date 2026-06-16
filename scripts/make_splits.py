"""Patient-level splits for ROP-Bench (Phase 1).

Outputs data/processed/patient_splits.csv: (dataset, patient_id, split).
- 70/15/15 train/val/test, split BY PATIENT (no image leakage), per dataset.
- Stratified by each patient's majority label (plus for FARFUM/Czech, else presence).
- HVDROPDB has no patient_id -> all rows = test (eval-only set).
This one table serves both designs:
  in-domain   : use train/val/test within a dataset.
  cross-site  : train on source dataset's train+val, test on ENTIRE target dataset.
"""
import os
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
SEED = 42
rng = np.random.default_rng(SEED)
M = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")

def patient_label(g):
    # stratify by whether the patient has ANY positive case — plus/ROP is a
    # per-series property, so majority-vote would hide the rare positives.
    for col in ("plus_label", "rop_presence"):
        v = g[col].dropna()
        if len(v):
            return int(v.max())
    return 0

out = []
for ds, g in M.groupby("dataset"):
    if g["patient_id"].isna().all():                      # HVDROPDB
        for pid in [None]:
            pass
        out.append(g.assign(split="test")[["dataset", "patient_id", "image_path"]]
                   .assign(split="test"))
        print(f"{ds:9s}: no patient_id -> {len(g)} imgs all TEST")
        continue
    pl = g.groupby("patient_id").apply(patient_label, include_groups=False)
    pats = pl.index.to_numpy()
    splitmap = {}
    for lab in sorted(pl.unique()):                       # stratify by patient label
        ps = rng.permutation(pats[pl.values == lab])
        n = len(ps); ntr, nva = int(.70 * n), int(.15 * n)
        for p in ps[:ntr]:        splitmap[p] = "train"
        for p in ps[ntr:ntr+nva]: splitmap[p] = "val"
        for p in ps[ntr+nva:]:    splitmap[p] = "test"
    rows = g[["dataset", "patient_id", "image_path"]].copy()
    rows["split"] = rows["patient_id"].map(splitmap)
    out.append(rows)
    npat = g["patient_id"].nunique()
    cnt = rows.groupby("split").patient_id.nunique()
    print(f"{ds:9s}: {npat} patients -> "
          f"train {cnt.get('train',0)} / val {cnt.get('val',0)} / test {cnt.get('test',0)}")

S = pd.concat(out, ignore_index=True)
dst = ROOT / "data" / "processed" / "patient_splits.csv"
dst.parent.mkdir(parents=True, exist_ok=True)
S.to_csv(dst, index=False)
print(f"\nwrote {len(S)} rows -> {dst}")
print("image split totals:", dict(S.split.value_counts()))
