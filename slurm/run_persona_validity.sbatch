#!/bin/bash
# [단계 20] 페르소나 조작 점검(32B 평가자). 사용법: sbatch slurm/run_persona_validity.sbatch
#SBATCH -J dmp-pvalid
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
.venv-gen/bin/python scripts/29_persona_validity.py
back logs/persona_validity.txt
