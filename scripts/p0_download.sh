#!/usr/bin/env bash
# Phase 0 bulk download: FARFUM (rar/patient) + Shenzhen (zip) + HVDROPDB (Mendeley)
set -u
PROJECT_ROOT="${ROP_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$PROJECT_ROOT/data/raw"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# --- FARFUM: 68 patient rars (labels already present) ---
mkdir -p "$ROOT/farfum/rars"
log "FARFUM: downloading rars from manifest..."
while IFS=$'\t' read -r name url size; do
  case "$name" in *.rar) out="$ROOT/farfum/rars/$name";;
                   *) continue;; esac          # skip xlsx (already have)
  [ -s "$out" ] && { log "  skip $name (exists)"; continue; }
  curl -sL "$url" -o "$out" && log "  ok $name ($((size/1000000))MB)" || log "  FAIL $name"
done < "$ROOT/farfum/manifest.tsv"

# --- Shenzhen ---
mkdir -p "$ROOT/shenzhen"
log "Shenzhen: downloading zip (203MB)..."
curl -sL "https://ndownloader.figshare.com/files/65156070" -o "$ROOT/shenzhen/ROP_dataset.zip" && log "  ok zip"
curl -sL "https://ndownloader.figshare.com/files/65156304" -o "$ROOT/shenzhen/zip_information.xlsx" && log "  ok xlsx"

# --- HVDROPDB (Mendeley v3) ---
mkdir -p "$ROOT/hvdropdb"
log "HVDROPDB: downloading classification rar (207MB) + segmentation zip (342MB)..."
curl -sL "https://data.mendeley.com/public-files/datasets/xw5xc7xrmp/files/872632e0-35d8-4262-b25a-051a92ac340d/file_downloaded" \
     -o "$ROOT/hvdropdb/HVDROPDB_RetCam_Neo_Classification.rar" && log "  ok classification rar"

log "ALL DOWNLOADS DONE"
du -sh "$ROOT"/*/ 2>/dev/null
