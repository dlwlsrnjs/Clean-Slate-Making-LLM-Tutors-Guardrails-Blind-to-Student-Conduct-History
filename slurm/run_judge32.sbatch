#!/bin/bash
# [단계 12] Qwen3-32B(v2 기준)로 검증용 200개를 판정하고 8B 판정과 비교한다. 사용법: sbatch slurm/run_judge32.sbatch
#SBATCH -J dmp-judge32
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
export HF_HOME=/resilience/ljk98/hf_cache
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
.venv-gen/bin/python scripts/12_judge.py --gens data/gens_valset.jsonl --out data/judged_valset_q32b.jsonl --judge Qwen/Qwen3-32B --rubric v2
mkdir -p /home/ljk98/dm-persona/logs/review
python3 -I scripts/16_compare_judges.py data q32b /home/ljk98/dm-persona/logs/review/judge32b_valset.md | tee logs/judge32b_valset.txt
cp logs/judge32b_valset.txt /home/ljk98/dm-persona/logs/
