#!/bin/bash
# [단계 15] 직접 검토 덤프 + 길이/문구/누출 통계. CPU job. 사용법: sbatch slurm/run_review_dpo3.sbatch "<base 태그 + 비교 태그들(마지막이 덤프 대상)>" <출력 이름 예: dpo4>
#SBATCH -J dmp-rev3
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
python3 -I scripts/19_review_dpo3.py logs/review_${2:-dpo3}.md ${1:-llama3 llama3_dpo1 llama3_dpo2 llama3_dpo3} | tee logs/review_${2:-dpo3}_stats.txt
mkdir -p /home/ljk98/dm-persona/logs/review
cp logs/review_${2:-dpo3}.md /home/ljk98/dm-persona/logs/review/
cp logs/review_${2:-dpo3}_stats.txt /home/ljk98/dm-persona/logs/
