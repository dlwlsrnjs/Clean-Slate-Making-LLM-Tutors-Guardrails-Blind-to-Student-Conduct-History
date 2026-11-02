#!/bin/bash
# [단계 22] CPC-DPO 학습. 사용법: sbatch slurm/run_cpc.sbatch <HF 모델 id> <쌍둥이 쌍 접두사> <출력 이름> [31 의 추가 인자: --lam 0.05 --seed 0 ...]
#SBATCH -J dmp-cpc
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
export HF_HOME=/resilience/ljk98/hf_cache
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
MODEL=$1; PAIRS=$2; OUT=$3; shift 3
.venv-train/bin/python scripts/31_train_cpcdpo.py --model "$MODEL" --pairs data/$PAIRS.jsonl --out runs/$OUT "$@"
ls -la runs/$OUT
