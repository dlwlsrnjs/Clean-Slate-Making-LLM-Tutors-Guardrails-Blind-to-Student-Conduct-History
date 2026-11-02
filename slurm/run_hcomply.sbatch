#!/bin/bash
# [단계 15-2] H 에서 comply 가 나온 쿼리 분포. CPU job. 사용법: sbatch slurm/run_hcomply.sbatch "<태그 목록>"
#SBATCH -J dmp-hcomp
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
python3 -I scripts/20_h_comply_by_query.py $1 | tee logs/h_comply.txt
cp logs/h_comply.txt /home/ljk98/dm-persona/logs/
