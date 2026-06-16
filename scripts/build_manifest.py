"""ROP-Bench unified manifest builder (Phase 1).

One row per image across all 4 datasets, with harmonized labels:
  plus_label   : 1=plus disease, 0=not-plus (normal/pre-plus). NA if dataset lacks plus.
  rop_presence : 1=active ROP, 0=no active ROP. NA where not derivable.
  stage_raw    : native staging string (kept verbatim).
  device, country, patient_id, eye, series_id, image_path.

Harmonization rules (documented; see notes/phase0_findings.md):
  FARFUM  plus: consensus Label 3->1, {1,2}->0.  rop_presence: NA (plus-only consensus).
  CZECH   plus: PF 2->1, 0->0 (no PF=1 exists).
          rop_presence: DG 0|1 ->0 ; DG 2..9 ->1 ; DG 10..13 (non-ROP path) -> NA (distractor).
  SHENZHEN rop_presence: Normal->0 ; Stage1/2/3->1 ; laser_scars-> NA (post-treatment). plus NA.
  HVDROPDB rop_presence: *_Normal->0 ; *_ROP->1. plus NA. (no patient id)
"""
import re, glob, os, pandas as pd, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT_ROOT / "data" / "raw"
rows = []

# ---------- FARFUM ----------
raw = pd.read_excel(f"{ROOT}/farfum/Dataset_Labels.xlsx", header=None)
fl = raw.iloc[2:, [0, 1, 17]].copy(); fl.columns = ["patient", "image", "Label"]
fl = fl[fl["image"].notna()]
lab = {str(r.image).strip(): r.Label for r in fl.itertuples()}
for f in glob.glob(f"{ROOT}/farfum/images/*/*.jpg"):
    stem = os.path.basename(f)[:-4]                 # uuid.frame
    pat = os.path.basename(os.path.dirname(f))
    L = lab.get(stem, np.nan)
    plus = (1 if L == 3 else 0) if pd.notna(L) else np.nan
    rows.append(dict(dataset="FARFUM", country="Iran", patient_id=f"farfum_{pat}",
                     eye=np.nan, series_id=np.nan, device="RetCam",
                     plus_label=plus, rop_presence=np.nan,
                     stage_raw=f"plusLabel={L}", image_path=f))

# ---------- CZECH ----------
DEV = {1: "ClarityRetCam3", 2: "PhoenixICON", 3: "NatusEnvision"}
for f in glob.glob(f"{ROOT}/czech/ext/images_stack/*.jpg"):
    m = re.match(r"(\d+)_._GA\d+_BW\d+_PA[\d.]+_DG(\d+)_PF(\d+)_D(\d+)_S(\d+)_",
                 os.path.basename(f))
    if not m:
        continue
    pid, dg, pf, d, s = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    plus = 1 if pf == 2 else 0
    pres = 0 if dg in (0, 1) else (1 if 2 <= dg <= 9 else np.nan)
    rows.append(dict(dataset="Czech", country="Czech", patient_id=f"czech_{pid}",
                     eye=np.nan, series_id=f"czech_{pid}_S{s}", device=DEV[d],
                     plus_label=plus, rop_presence=pres,
                     stage_raw=f"DG{dg}", image_path=f))

# ---------- SHENZHEN ----------
zi = pd.read_excel(f"{ROOT}/shenzhen/zip_information.xlsx")
pid_map = {str(r.img_name).strip(): (r.ID, r.eye) for r in zi.itertuples()}
PRES = {"Normal": 0, "Stage1": 1, "Stage2": 1, "Stage3": 1, "laser scars": np.nan}
for d in glob.glob(f"{ROOT}/shenzhen/ext/ROP dataset/image/*"):
    cls = os.path.basename(d)
    for f in glob.glob(d + "/*"):
        bn = os.path.basename(f)
        pid, eye = pid_map.get(bn, (np.nan, np.nan))
        rows.append(dict(dataset="Shenzhen", country="China",
                         patient_id=f"shenzhen_{pid}" if pd.notna(pid) else np.nan,
                         eye=eye, series_id=np.nan, device="mixed",
                         plus_label=np.nan, rop_presence=PRES.get(cls, np.nan),
                         stage_raw=cls, image_path=f))

# ---------- HVDROPDB ----------
for d in glob.glob(f"{ROOT}/hvdropdb/ext/*"):
    name = os.path.basename(d)            # e.g. RetCam_ROP
    dev, cls = name.split("_", 1)
    pres = 1 if cls.lower() == "rop" else 0
    for f in glob.glob(d + "/*.png") + glob.glob(d + "/*.jpg"):
        rows.append(dict(dataset="HVDROPDB", country="India", patient_id=np.nan,
                         eye=np.nan, series_id=np.nan, device=dev,
                         plus_label=np.nan, rop_presence=pres,
                         stage_raw=cls, image_path=f))

df = pd.DataFrame(rows)
out = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"manifest rows: {len(df)} -> {out}\n")

print("=== PLUS task (plus vs not) ===")
p = df[df.plus_label.notna()]
print(p.groupby(["dataset", "country"]).plus_label.agg(
    n="size", plus="sum").assign(plus=lambda x: x.plus.astype(int)).to_string())
print(f"  TOTAL plus-labeled: {len(p)}  (plus={int(p.plus_label.sum())})")
print(f"  patients: {p.patient_id.nunique()}")

print("\n=== ROP-PRESENCE task (active ROP vs not) ===")
r = df[df.rop_presence.notna()]
print(r.groupby(["dataset", "country"]).rop_presence.agg(
    n="size", rop="sum").assign(rop=lambda x: x.rop.astype(int)).to_string())
print(f"  TOTAL presence-labeled: {len(r)}  (rop={int(r.rop_presence.sum())})")

print("\n=== CROSS-DEVICE coverage ===")
print(df.groupby(["dataset", "device"]).size().to_string())

print("\n=== data hygiene ===")
print("  rows missing patient_id:", int(df.patient_id.isna().sum()),
      "(", dict(df[df.patient_id.isna()].dataset.value_counts()), ")")
