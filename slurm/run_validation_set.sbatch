#!/bin/bash
# [단계 9] 판정자 검증용 층화 200개 파일 생성. CPU 만 쓰는 짧은 job. 사용법: sbatch slurm/run_validation_set.sbatch
#SBATCH -J dmp-valset
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
python3 -I scripts/07_make_validation_set.py data 200 --run llama3 --run qwen3 --run mistral --run llama3_dpo1
# 로그인 노드에서 볼 수 있게 /home 에도 복사 (200행이라 작다)
cp data/judge_validation_200.csv data/judge_validation_key.jsonl /home/ljk98/dm-persona/data/
