"""Cross-DEVICE transfer within Czech (population-controlled) — presence task.

Czech images carry a per-image device tag (3 machines), same patient population,
so train-on-device-A / test-on-device-B isolates DEVICE shift from country/population
shift. Off-diagonal = cross-device (train all A, test all B). Diagonal = in-device
(patient-level 70/30 split). Bootstrap CIs by patient on the test device.

Usage: python cross_device.py --backbone {vit_base_patch14_dinov2.lvd142m_224,retfound_cfp,...}
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
B, rng = 2000, np.random.default_rng(0)
DEVS = ["ClarityRetCam3", "PhoenixICON", "NatusEnvision"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m_224")
    a = ap.parse_args()
    M = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    c = M[(M.dataset == "Czech") & M.rop_presence.notna()].copy()
    z = np.load(ROOT / "data" / "processed" / "feats" / a.backbone / "Czech.npz", allow_pickle=True)
    feat = dict(zip(z["paths"], z["X"]))
    c["X"] = list(c.image_path.map(feat))

    def Xy(df):
        return np.stack(df.X.values), df.rop_presence.to_numpy().astype(int)

    def boot(y, s, grp):
        u = np.unique(grp); idxby = {g: np.where(grp == g)[0] for g in u}; out = []
        for _ in range(B):
            idx = np.concatenate([idxby[g] for g in rng.choice(u, len(u), replace=True)])
            if len(set(y[idx])) > 1:
                out.append(roc_auc_score(y[idx], s[idx]))
        return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)

    hdr = "train\\test".rjust(16)
    print(f"backbone={a.backbone}")
    print(hdr + "".join(f"{d[:10]:>22s}" for d in DEVS))
    for src in DEVS:
        row = f"{src[:14]:>16s}"
        gsrc = c[c.device == src]
        for tgt in DEVS:
            gtgt = c[c.device == tgt]
            if src == tgt:                                   # in-device 70/30 by patient
                pats = gsrc.patient_id.unique(); rng2 = np.random.default_rng(1)
                tr_p = set(rng2.choice(pats, int(.7 * len(pats)), replace=False))
                tr = gsrc[gsrc.patient_id.isin(tr_p)]; te = gsrc[~gsrc.patient_id.isin(tr_p)]
            else:
                tr, te = gsrc, gtgt
            Xtr, ytr = Xy(tr); Xte, yte = Xy(te)
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                row += f"{'n/a':>22s}"; continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
            s = clf.predict_proba(Xte)[:, 1]
            m, lo, hi = boot(yte, s, te.patient_id.to_numpy())
            tag = "*" if src == tgt else " "
            row += f"{m:.2f}[{lo:.2f},{hi:.2f}]{tag:>1s}".rjust(22)
        print(row)

if __name__ == "__main__":
    main()
