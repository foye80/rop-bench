"""Phase 0 — FARFUM-RoP label schema & inter-grader agreement.
Two-row merged header; assign columns by position (18 cols, 5 graders A-E)."""
import pandas as pd, numpy as np, sys

raw = pd.read_excel(sys.argv[1] if len(sys.argv) > 1
                    else "data/raw/farfum/Dataset_Labels.xlsx", header=None)
graders = list("ABCDE")
cols = ["patient", "image"]
for g in graders:
    cols += [f"grade_{g}", f"stage_{g}", f"diag_{g}"]
cols += ["Label"]
df = raw.iloc[2:].copy()
df.columns = cols
df["patient"] = df["patient"].ffill()
df = df[df["image"].notna()].reset_index(drop=True)

gcols = [f"grade_{g}" for g in graders]
G = df[gcols].apply(pd.to_numeric, errors="coerce")
# grade==0 means "not specified by this grader" -> treat as abstention/missing.
# grade 1/2/3 == Normal/Pre-Plus/Plus (same scale as consensus Label).
G = G.replace(0, np.nan)
L = pd.to_numeric(df["Label"], errors="coerce")
LABNAME = {1: "Normal", 2: "Pre-Plus", 3: "Plus"}

print(f"images={len(df)}  patients={df['patient'].nunique()}")
print("consensus Label dist:",
      {LABNAME[k]: int(v) for k, v in L.value_counts().sort_index().items()})
print("\n--- how many of the 5 graders actually labeled each image ---")
ng = G.notna().sum(axis=1)
print("graders-per-image dist:", dict(ng.value_counts().sort_index()))
for g in graders:
    vc = G[f"grade_{g}"].value_counts().sort_index()
    print(f"  grader {g}: labeled {G[f'grade_{g}'].notna().sum():4d} imgs, dist {dict(vc)}")

print("\n--- inter-grader agreement on labeled subset (1/2/3) ---")
ge2 = G[ng >= 2]                       # images with >=2 graders
unan = ge2.apply(lambda r: r.dropna().nunique() == 1, axis=1).mean()
print(f"images with >=2 graders: {len(ge2)}; of those, unanimous: {unan:.1%}")

def fleiss_kappa(M):
    N = len(M); n = M.sum(1)
    p = M.sum(0) / M.sum()
    P = ((M ** 2).sum(1) - n) / (n * (n - 1))
    return (P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())

cats = [1, 2, 3]
rows = [[np.sum(r == c) for c in cats] for r in G.values
        if sum(np.sum(r == c) for c in cats) >= 2]
M = np.array(rows)
print(f"Fleiss kappa (>=2 graders, n={len(M)}): {fleiss_kappa(M):.3f}")
print("NOTE: no image has all 5 graders; max graders/image = 3.")

# Diagnostic (clinical action)
dcols = [f"diag_{g}" for g in graders]
D = df[dcols].replace({"teatment": "treatment"})
print("\nDiagnostic values:", sorted(set(D.values.ravel()) - {np.nan},
                                    key=str))
any_treat = D.eq("treatment").any(axis=1).sum()
all_treat = D.eq("treatment").all(axis=1).sum()
print(f"images where ANY grader says treatment: {any_treat}")
print(f"images where ALL graders say treatment: {all_treat}")
