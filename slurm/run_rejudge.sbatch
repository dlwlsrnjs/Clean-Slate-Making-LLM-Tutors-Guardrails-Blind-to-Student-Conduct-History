#!/bin/bash
# [단계 12-재판정] 생성물을 32B 판정자(v2.1 기준) + 재정의 gold(labels_v2.json)로 다시 판정한다.
# 사용법: sbatch slurm/run_rejudge.sbatch "llama3_train llama3 llama3_dpo1 llama3_dpo2"   (공백으로 구분한 tag 목록)
#SBATCH -J dmp-rejudge
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
fetch data/labels_v2.json
for tag in ${1:-llama3_train llama3 llama3_dpo1 llama3_dpo2}; do
  if [ ! -s data/gens_$tag.jsonl ]; then echo "skip $tag (no gens)"; continue; fi
  .venv-gen/bin/python scripts/12_judge.py --gens data/gens_$tag.jsonl --out data/judged_${tag}_q32.jsonl \
      --judge Qwen/Qwen3-32B --rubric v2 --labels data/labels_v2.json
done
