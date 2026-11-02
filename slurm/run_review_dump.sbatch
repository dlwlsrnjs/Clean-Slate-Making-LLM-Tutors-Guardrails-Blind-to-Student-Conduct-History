#!/bin/bash
# [단계 11] 직접 읽고 점검할 표본(판정·DPO 쌍·전/후 응답)을 /home/ljk98/dm-persona/logs/review/ 에 마크다운으로 만든다. CPU job.
# 사용법: sbatch slurm/run_review_dump.sbatch <기준 태그> <비교 태그> <쌍 접두사> [출력 하위폴더]
#SBATCH -J dmp-review
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
python3 -I scripts/14_review_dump.py data /home/ljk98/dm-persona/logs/review${4:+/$4} --base "$1" --new "$2" --pairs "$3"
