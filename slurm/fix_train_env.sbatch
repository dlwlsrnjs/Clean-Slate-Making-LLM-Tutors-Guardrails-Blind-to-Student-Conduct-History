#!/bin/bash
# .venv-train 보정: TRL 0.11 은 설치된 transformers 와 시그니처가 안 맞아 0.19.1 로 올리고, pyarrow 는 datasets 호환을 위해 낮춘다.
#SBATCH -J dmp-fixenv
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
cd /resilience/ljk98/dm-persona
export UV_CACHE_DIR=/resilience/ljk98/uv_cache HF_HOME=/resilience/ljk98/hf_cache
uv pip install --python .venv-train/bin/python "pyarrow<21" "trl==0.19.1"
.venv-train/bin/python -c "import pyarrow, datasets, trl; from trl import DPOConfig, DPOTrainer; print(pyarrow.__version__, datasets.__version__, trl.__version__, \"DPOTrainer ok\")"
