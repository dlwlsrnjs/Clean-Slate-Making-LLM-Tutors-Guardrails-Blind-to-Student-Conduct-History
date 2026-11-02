#!/bin/bash
# [단계 15-2] H 에서 comply 가 나온 쿼리 분포. CPU job. 사용법: sbatch slurm/run_hcomply.sbatch "<태그 목록>"
#SBATCH -J dmp-hcomp
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
python3 -I scripts/20_h_comply_by_query.py $1 | tee logs/h_comply.txt
back logs/h_comply.txt
