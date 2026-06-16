"""Phase 0 — combined per-class image counts across all 4 datasets."""
import re, glob, os, collections, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ROP_BENCH_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT_ROOT / "data" / "raw"

print("="*70, "\nCZECH (Ostrava) — parsed from filenames (per-image)")
PF = {0: "Normal", 1: "Pre-Plus", 2: "Plus"}
DEV = {1: "ClarityRetCam3", 2: "PhoenixICON", 3: "NatusEnvision"}
pf, dg, dev, pat = collections.Counter(), collections.Counter(), collections.Counter(), set()
for f in glob.glob(f"{ROOT}/czech/ext/images_stack/*.jpg"):
    m = re.match(r"(\d+)_._GA\d+_BW\d+_PA[\d.]+_DG(\d+)_PF(\d+)_D(\d+)_S", os.path.basename(f))
    if not m:
        print("  UNPARSED:", os.path.basename(f)); continue
    pat.add(m.group(1)); dg[m.group(2)] += 1
    pf[int(m.group(3))] += 1; dev[int(m.group(4))] += 1
print(f"  images={sum(pf.values())}  patients={len(pat)}")
print("  PLUS FORM:", {PF[k]: pf[k] for k in sorted(pf)})
print("  DIAGNOSIS code:", dict(sorted(dg.items())))
print("  DEVICE:", {DEV[k]: dev[k] for k in sorted(dev)})

print("="*70, "\nFARFUM — consensus plus label (from xlsx)")
raw = pd.read_excel(f"{ROOT}/farfum/Dataset_Labels.xlsx", header=None)
L = pd.to_numeric(raw.iloc[2:, 17], errors="coerce").dropna()
print("  PLUS:", {("Normal","Pre-Plus","Plus")[int(k)-1]: int(v)
                  for k, v in L.value_counts().sort_index().items()})

print("="*70, "\nSHENZHEN — folder=stage")
for d in sorted(glob.glob(f"{ROOT}/shenzhen/ext/ROP dataset/image/*")):
    n = len(glob.glob(d+"/*.jpg")) + len(glob.glob(d+"/*.png")) + len(glob.glob(d+"/*.jpeg"))
    print(f"  {os.path.basename(d):16s} {n}")

print("="*70, "\nHVDROPDB — folder = device_class")
for d in sorted(glob.glob(f"{ROOT}/hvdropdb/ext/*")):
    n = len(glob.glob(d+"/*.png"))+len(glob.glob(d+"/*.jpg"))
    print(f"  {os.path.basename(d):16s} {n}")
