#!/bin/bash
# 모델 가중치를 $HF_HOME(기본 $DMP_WORK/hf_cache) 로 내려받는다 (CPU job). 사용법: sbatch slurm/download_model.sbatch <HF 모델 id>
#SBATCH -J dmp-download
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
.venv-gen/bin/python -c "
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
print('done', p)" "$1"
du -sh $HF_HOME/hub/models--$(echo "$1" | sed 's#/#--#')
