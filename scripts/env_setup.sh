#!/usr/bin/env bash
# Build dedicated env at project-local path (self-contained, on scratch).
set -e
PROJECT_ROOT="${ROP_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV="${ROP_BENCH_ENV:-$PROJECT_ROOT/env}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.pip_cache}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "[1/3] conda create -> $ENV"
/opt/conda/condabin/conda create --prefix "$ENV" python=3.11 -y -q

PY="$ENV/bin/python"
echo "[2/3] torch (cu124) ..."
"$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[3/3] ML/eval deps ..."
"$PY" -m pip install -q timm huggingface_hub safetensors scikit-learn \
      pandas numpy pillow tqdm matplotlib open_clip_torch openpyxl

echo "=== verify ==="
"$PY" - <<'EOF'
import torch, timm, sklearn, pandas
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("timm", timm.__version__, "| sklearn", sklearn.__version__)
print("cuda available:", torch.cuda.is_available(),
      "| device count:", torch.cuda.device_count())
EOF
echo "ENV_SETUP_DONE"
