#!/bin/bash
# [단계 7] LoRA DPO 학습. 사용법: sbatch slurm/run_dpo.sbatch <HF 모델 id> <쌍 접두사 예: dpo_llama3> <출력 이름 예: dpo1> [04 의 추가 인자...]
# 결과 어댑터: /resilience/ljk98/dm-persona/runs/<출력 이름>  → run_probe.sbatch 의 LoRA 인자로 쓴다.
#SBATCH -J dmp-dpo
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
export HF_HOME=/resilience/ljk98/hf_cache
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
MODEL=$1; PAIRS=$2; OUT=$3; shift 3
.venv-train/bin/python scripts/04_train_dpo.py --model "$MODEL" --pairs data/$PAIRS.jsonl --out runs/$OUT "$@"
ls -la runs/$OUT
