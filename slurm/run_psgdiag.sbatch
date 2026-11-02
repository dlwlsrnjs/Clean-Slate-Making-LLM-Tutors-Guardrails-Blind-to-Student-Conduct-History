#!/bin/bash
# [단계 16] PSG 진단(A1~A6): 중립 대비 분해·분산 분해·키워드 분류기·조건별 범위·BH·쿼리별 PSG. CPU job.
# 사용법: sbatch slurm/run_psgdiag.sbatch "<태그 목록>" [출력 이름]   예: "llama3 llama3_dpo3 llama3_dpo4 llama3_dpo5" psgdiag
#SBATCH -J dmp-psgdiag
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
OUT=logs/${2:-psgdiag}.txt
.venv-gen/bin/python -I scripts/25_psg_diagnostics.py $1 | tee "$OUT"
cp "$OUT" /home/ljk98/dm-persona/logs/
