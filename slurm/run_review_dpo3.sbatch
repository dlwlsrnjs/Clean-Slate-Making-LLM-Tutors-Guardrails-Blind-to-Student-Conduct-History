#!/bin/bash
# [단계 15] 직접 검토 덤프 + 길이/문구/누출 통계. CPU job. 사용법: sbatch slurm/run_review_dpo3.sbatch "<base 태그 + 비교 태그들(마지막이 덤프 대상)>" <출력 이름 예: dpo4>
#SBATCH -J dmp-rev3
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
python3 -I scripts/19_review_dpo3.py logs/review_${2:-dpo3}.md ${1:-llama3 llama3_dpo1 llama3_dpo2 llama3_dpo3} | tee logs/review_${2:-dpo3}_stats.txt
mkdir -p logs/review
cp logs/review_${2:-dpo3}.md logs/review/
back logs/review/review_${2:-dpo3}.md logs/review_${2:-dpo3}_stats.txt
