#!/bin/bash
# [단계 11-2] 감사 수치. CPU job. 사용법: sbatch slurm/run_audit_stats.sbatch
#SBATCH -J dmp-audit
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
python3 -I scripts/15_audit_stats.py data dpo_llama3 dpo_llama3_v2 --runs llama3 llama3_dpo1 llama3_dpo2 | tee logs/audit_stats.txt
cp logs/audit_stats.txt /home/ljk98/dm-persona/logs/
