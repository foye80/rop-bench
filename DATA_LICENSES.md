# Data Sources and License Notes

Raw fundus images are not redistributed in this repository. Users must obtain
the source datasets from their original public sources and follow the license or
terms attached to each dataset.

## Upstream Datasets

1. FARFUM-RoP

- Dataset paper: Akbari et al., Scientific Data, 2024.
- DOI: https://doi.org/10.1038/s41597-024-03897-7
- Used here for plus disease labels and sparse grader-specific labels.

2. Shenzhen Eye Hospital ROP dataset

- Dataset paper: Zhao et al., Scientific Data, 2024.
- DOI: https://doi.org/10.1038/s41597-024-03362-5
- Used here for active ROP presence based on Normal versus Stage 1-3 labels.

3. HVDROPDB

- Dataset paper: Agrawal et al., Data in Brief, 2024.
- DOI: https://doi.org/10.1016/j.dib.2023.109839
- Used here only as an external test set because patient identifiers are not
  available.

4. Ostrava Retinal Image Dataset of Infants and ROP

- Dataset paper: Timkovic et al., Scientific Data, 2024.
- DOI: https://doi.org/10.1038/s41597-024-03409-7
- Used here for active ROP presence, plus disease labels and device metadata.

## Released Files

The following derived files are released:

- `data/processed/manifest_public.csv`
- `data/processed/patient_splits_public.csv`
- `data/processed/finetune_results.csv`
- manuscript figures and analysis scripts

These files do not contain raw retinal images. They contain harmonized labels,
dataset names, patient IDs as provided or derived from public dataset structure,
and stable source image identifiers. The `source_image_id` field is relative to
the expected `data/raw/` layout and includes enough dataset subdirectory context
to distinguish files with the same basename.

If an upstream dataset license imposes additional requirements, those terms take
priority for the corresponding raw images.
