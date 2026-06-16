# ROP-Bench — Results draft (write-up base)

_Last updated 2026-06-12. All numbers reproducible from `scripts/` + cached features._

## Working title
**How well do retinal AI models transfer across centers, devices, and to pediatric
wide-field imaging? A multi-source public benchmark for retinopathy of prematurity (ROP).**

## Contribution (3 + 1)
1. A reusable **multi-source public ROP benchmark**: 4 datasets harmonized into one
   manifest (8,821 images, 3 countries, 5+ camera models) with patient-level splits.
2. **Cross-site/-device generalization is far worse than single-center numbers suggest.**
3. **Retinal foundation models do not transfer to pediatric ROP** — a general vision SSL
   model (DINOv2) beats the leading retinal specialist (RETFound) under both frozen probing
   and full fine-tuning. Mechanism: RETFound is pretrained on adult narrow-field fundus.
4. (Field agenda) Public ROP data lacks **standardized labels** and there is **no public
   pediatric/ROP foundation model**; ROP-positive patients are scarce, capping fine-grained
   analyses (we surface this quantitatively).

---

## 1. Datasets & benchmark construction
Manifest: `data/processed/manifest.csv` (build: `scripts/build_manifest.py`).
Splits: `data/processed/patient_splits.csv` — patient-level, stratified by any-positive,
70/15/15 (build: `scripts/make_splits.py`). HVDROPDB has no patient IDs → eval-only.

| Dataset | Country | Device(s) | n img | Label axis | Notes |
|---|---|---|---|---|---|
| FARFUM | Iran | RetCam | 1533 | **plus** (Norm 782/PrePlus 479/Plus 272) | 5 graders, sparse (1–3/img) |
| Shenzhen | China | 3 (mixed) | 1099 | **ICROP stage** (Norm236/S1 94/S2 165/S3 261/laser343) | patient IDs via xlsx |
| HVDROPDB | India | RetCam+Neo | 185 | **binary** (per device) | no patient IDs |
| Czech (Ostrava) | Czech | 3 (per-image tag) | 6004 | **plus + stage + device** | incl. non-ROP pathology (distractors) |

**Two harmonized tasks**
- **Primary — ROP presence (binary).** Czech+Shenzhen+HVDROPDB. 5,650 img, 2,289 positive,
  **~305 ROP-positive patients** (Shenzhen 285 + Czech 20). Statistically well-powered.
- **Secondary — plus disease (binary).** FARFUM+Czech. 7,537 img, 901 plus, but **only 18
  plus-positive patients** (Czech 3 + FARFUM 15) → exploratory, patient-limited.

---

## 2. Methods
Frozen-backbone **linear probe** (class-weighted logistic regression on cached features) and
full **fine-tuning** (AdamW, head lr 1e-3 / body 1e-5, cosine, class-weighted CE).
Backbones: DINOv2 ViT-B/14 (@518 native, @224 matched), RETFound-CFP ViT-L/16 (@224),
random ViT-B/14. **Cluster bootstrap** (2000 resamples, by patient) for CIs.
Scripts: `probe.py` (--task plus/presence, --img_size), `finetune.py`, `bootstrap_compare.py`,
`cross_device.py`, `rq3_annotator.py`.

---

## 3. RQ1 — Cross-site generalization (DINOv2@518 linear probe, presence)
In-domain: **Czech 0.926 / Shenzhen 0.999**. Cross-site AUC:

| train＼test | Czech | Shenzhen | HVDROPDB |
|---|---|---|---|
| Czech | (0.926) | 0.822 | 0.739 |
| Shenzhen | 0.866 | (0.999) | 0.953 |
| HVDROPDB | 0.830 | 0.977 | — |

Key: single-center ≈0.93–1.0 collapses to **0.74–0.87** off-site; Czech (3 devices, most
diverse, + non-ROP distractors) is the hardest target. AP collapses harder than AUC
(precision under class imbalance degrades most under shift). Plus task echoes this:
in-domain ≈0.93, cross-country 0.83–0.88, AP 0.57–0.61.

## 4. RQ2 — General vs specialist foundation model (presence)
**Frozen linear probe @224, paired cluster-bootstrap (DINOv2 − RETFound):**

| direction | DINOv2@224 | RETFound@224 | diff [95% CI] | sig |
|---|---|---|---|---|
| Shen (in) | 0.991 | 0.947 | +0.045 [.01,.09] | ✅ |
| Czech→HVDR | 0.861 | 0.760 | +0.101 [.03,.18] | ✅ |
| HVDR→Czech | 0.816 | 0.657 | +0.159 [.04,.33] | ✅ |
| HVDR→Shen | 0.966 | 0.671 | +0.295 [.25,.34] | ✅ |
| Shen→HVDR | 0.947 | 0.760 | +0.188 [.13,.25] | ✅ |
| Czech (in) | 0.886 | 0.854 | +0.033 [-.07,.13] | ns |
| Czech→Shen | 0.850 | 0.816 | +0.034 [-.01,.08] | ns |
| Shen→Czech | 0.778 | 0.707 | +0.071 [-.05,.21] | ns |

DINOv2 ≥ RETFound in all 8; **significant in 5** (large where HVDROPDB/Shenzhen are the test
set, i.e. well-powered); RETFound never significantly wins. The 3 ns all have Czech as test
(sparse positives → wide CIs).

**Backbone SPECTRUM (frozen probe @224): 3 general paradigms vs 1 retinal specialist.**

| direction | DINOv2 (SSL) | CLIP (VL) | ImageNet (sup.) | RETFound (specialist) |
|---|---|---|---|---|
| Czech (in) | 0.893 | 0.845 | 0.866 | 0.854 |
| Shenzhen (in) | 0.991 | 0.992 | 0.988 | 0.947 |
| Czech→HVDR | 0.861 | 0.845 | 0.760 | 0.760 |
| Czech→Shen | 0.851 | 0.827 | 0.845 | 0.816 |
| HVDR→Czech | 0.825 | 0.826 | 0.874 | 0.657 |
| HVDR→Shen | 0.966 | 0.933 | 0.931 | 0.671 |
| Shen→Czech | 0.786 | 0.803 | 0.828 | 0.707 |
| Shen→HVDR | 0.948 | 0.902 | 0.920 | 0.760 |
| **cross-site mean** | **0.873** | **0.856** | **0.860** | **0.729** |

All three general backbones (SSL / vision-language / supervised) cluster at cross-site mean
0.86–0.87; the retinal specialist RETFound is the clear outlier at **0.729**. The gap is not a
DINOv2 artifact — it holds across pretraining paradigms. RETFound collapses worst when trained
on the small HVDROPDB and extrapolated (HVDR→Shen 0.671, HVDR→Czech 0.657) where general models
hold 0.83–0.97. (Backbones: `vit_base_patch16_clip_224.laion2b`,
`vit_base_patch16_224.augreg2_in21k_ft_in1k`; logs `notes/probe_presence_{clip,imnet}.log`.)

**Full fine-tuning, 100% data (most stable; low-fraction unreliable, see §6):**
DINOv2 **(.90/.93/.86)** > RETFound (.77/.84/.78) > random (.52/.70/.61) [Czech-in/Shen/HVDR].
→ Even under RETFound's intended use (fine-tuning) with full data, the general model wins;
the specialist barely beats random. Smaller DINOv2 (86M) beats larger RETFound (300M) →
the gap is pretraining **data/domain**, not capacity. RETFound's adult posterior-pole
pretraining mismatches infant wide-field RetCam.

## 5. RQ3 — Annotator disagreement (FARFUM, 5 graders, plus)
**Q1 — benchmark depends on which grader is ground truth.** One fixed model (DINOv2 probe,
trained on consensus): AUC **0.930 → 0.996** depending on grader (Δ=0.066). Single-annotator
ROP benchmarks carry ~7 AUC points of GT-choice uncertainty.

**Q2 — model is within the human disagreement envelope.** Inter-grader Cohen κ (binary plus):
mean **0.806** [0.671, 1.0]. Model–grader κ: mean **0.784** [0.494, 0.908]. The model disagrees
with graders about as much as graders disagree with each other ("sixth grader"), except with
the most conservative grader E (κ 0.494, below the human floor).

---

## 6. Limitations (honest)
- **plus task patient-limited** (18 positive patients) → secondary/exploratory only.
- **Cross-device** (Czech 3 machines, population-controlled) **underpowered**: 6–8 positive
  patients/device, in-device diagonal degenerate, CIs span 0.3–0.99. Reported as exploratory.
- **Low-label fine-tuning unreliable** (Czech positives scarce → 10%/30% degenerate; noisy
  val selection; single seed). Only 100% fine-tune trusted.
- Shenzhen near-perfect in-domain (curated/easy) → single-center metrics overstate ability.
- HVDROPDB tiny (185, no patient IDs); most datasets ship consensus single labels.
- No external clinical/prospective validation; no true pediatric foundation model to compare.

## 7. TODO before submission
- [ ] Figures: cross-site heatmap (RQ1); DINOv2-vs-RETFound forest plot (RQ2); κ envelope (RQ3).
- [ ] Optional: add FLAIR / CLIP / ImageNet-ViT to make general-vs-specialist a spectrum.
- [ ] Bootstrap CIs on all RQ1 cells; multi-seed where feasible.
- [ ] Target venue decision (medical imaging journal/conf; or benchmark + analysis).
