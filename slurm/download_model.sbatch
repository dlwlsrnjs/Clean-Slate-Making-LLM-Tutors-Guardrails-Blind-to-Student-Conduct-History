#!/bin/bash
# 모델 가중치를 /resilience/ljk98/hf_cache 로 내려받는다 (CPU job). 사용법: sbatch slurm/download_model.sbatch <HF 모델 id>
#SBATCH -J dmp-download
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
export HF_HOME=/resilience/ljk98/hf_cache
cd /resilience/ljk98/dm-persona
.venv-gen/bin/python -c "
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
print('done', p)" "$1"
du -sh $HF_HOME/hub/models--$(echo "$1" | sed 's#/#--#')
