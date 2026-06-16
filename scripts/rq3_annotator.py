"""RQ3 — annotator disagreement (FARFUM, 5 graders) on PLUS disease.

Two questions:
 (1) Does the benchmark number depend on WHICH grader is ground truth?
     -> one fixed model (DINOv2 probe, trained on consensus), AUC vs each grader.
 (2) Is the model's disagreement with graders within the HUMAN disagreement envelope?
     -> Cohen kappa(model, grader_i) vs kappa(grader_i, grader_j).
plus_binary = (grade == 3). grade 0 = abstention -> excluded per grader.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, cohen_kappa_score

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
G = list("ABCDE")

# --- parse per-grader grades + consensus ---
raw = pd.read_excel(ROOT / "data" / "raw" / "farfum" / "Dataset_Labels.xlsx", header=None)
cols = ["patient", "image"] + sum([[f"grade_{g}", f"stage_{g}", f"diag_{g}"] for g in G], []) + ["Label"]
df = raw.iloc[2:].copy(); df.columns = cols
df = df[df.image.notna()].reset_index(drop=True)
for g in G:
    v = pd.to_numeric(df[f"grade_{g}"], errors="coerce").replace(0, np.nan)
    df[f"plus_{g}"] = (v == 3).astype(float).where(v.notna())     # 1=plus,0=not,nan=abstain
df["cons_plus"] = (pd.to_numeric(df.Label, errors="coerce") == 3).astype(int)

# --- features + split ---
z = np.load(ROOT / "data" / "processed" / "feats" / "vit_base_patch14_dinov2.lvd142m" / "FARFUM.npz", allow_pickle=True)
feat = {p.split("/")[-1][:-4]: x for p, x in zip(z["paths"], z["X"])}   # stem -> vec
M = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv").merge(
    pd.read_csv(ROOT / "data" / "processed" / "patient_splits.csv")[["image_path", "split"]], on="image_path")
M = M[M.dataset == "FARFUM"].copy()
M["stem"] = M.image_path.map(lambda p: p.split("/")[-1][:-4])
split = dict(zip(M.stem, M.split))
df["split"] = df.image.map(split)
df["X"] = df.image.map(lambda s: feat.get(s))
df = df[df.X.notna()].reset_index(drop=True)

# --- train one model on consensus (train split) ---
tr = df[df.split == "train"]; te = df[df.split == "test"]
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
    np.stack(tr.X.values), tr.cons_plus.values)
te = te.copy(); te["score"] = clf.predict_proba(np.stack(te.X.values))[:, 1]
thr = np.quantile(te.score, 1 - te.cons_plus.mean())          # match prevalence
te["pred"] = (te.score >= thr).astype(int)

print("=== Q1: same model, AUC vs each grader as ground-truth (TEST set) ===")
print(f"  vs CONSENSUS : AUC {roc_auc_score(te.cons_plus, te.score):.3f}  (n={len(te)}, plus={int(te.cons_plus.sum())})")
aucs = []
for g in G:
    sub = te[te[f"plus_{g}"].notna()]
    if sub[f"plus_{g}"].nunique() > 1:
        a = roc_auc_score(sub[f"plus_{g}"], sub.score); aucs.append(a)
        print(f"  vs grader {g}  : AUC {a:.3f}  (n={len(sub)}, plus={int(sub[f'plus_{g}'].sum())})")
print(f"  --> AUC SPREAD across graders: {min(aucs):.3f} - {max(aucs):.3f}  (Δ={max(aucs)-min(aucs):.3f})")

print("\n=== Q2: disagreement envelope (Cohen kappa, all labeled images) ===")
hh = []
for a, b in combinations(G, 2):
    m = df[df[f"plus_{a}"].notna() & df[f"plus_{b}"].notna()]
    if len(m) > 20 and m[f"plus_{a}"].nunique() > 1 and m[f"plus_{b}"].nunique() > 1:
        hh.append(cohen_kappa_score(m[f"plus_{a}"], m[f"plus_{b}"]))
print(f"  human-human kappa: mean {np.mean(hh):.3f}, range [{min(hh):.3f}, {max(hh):.3f}] (n_pairs={len(hh)})")
allp = df.copy(); allp["pred"] = (clf.predict_proba(np.stack(allp.X.values))[:, 1] >= thr).astype(int)
mh = []
for g in G:
    m = allp[allp[f"plus_{g}"].notna()]
    if m[f"plus_{g}"].nunique() > 1:
        k = cohen_kappa_score(m[f"plus_{g}"], m["pred"]); mh.append(k)
        print(f"  model vs grader {g}: kappa {k:.3f}")
print(f"  model-human kappa: mean {np.mean(mh):.3f}, range [{min(mh):.3f}, {max(mh):.3f}]")
print(f"\n  VERDICT: model-human mean {np.mean(mh):.3f} vs human-human mean {np.mean(hh):.3f} "
      f"-> model {'WITHIN' if np.mean(mh) >= min(hh) else 'BELOW'} human disagreement envelope")
