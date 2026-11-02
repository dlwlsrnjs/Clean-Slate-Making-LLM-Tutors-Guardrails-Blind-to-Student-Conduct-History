#!/bin/bash
# [단계 13] H·AM-H 쿼리 gold 재정의 (Qwen3-32B). 결과: data/labels_v2.json (+ 읽기용 사본을 /home 의 data/ 에)
#SBATCH -J dmp-relabel
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
.venv-gen/bin/python scripts/17_relabel_gold.py data/queries.jsonl data/labels_v2.json
back data/labels_v2.json
