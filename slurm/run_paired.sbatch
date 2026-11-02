#!/bin/bash
# [단계 18] 두 모델 PSG 차이의 쌍대 부트스트랩 검정. CPU job.
# 사용법: sbatch slurm/run_paired.sbatch "<A:B ...>" [출력 이름]   예: "llama3_ctrl5:llama3_dpo5 llama3:llama3_ctrl5" paired
#SBATCH -J dmp-paired
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
OUT=logs/${2:-paired}.txt
.venv-gen/bin/python -I scripts/27_paired_contrast.py $1 | tee "$OUT"
back "$OUT"
