#!/bin/bash
# [단계 21] 쌍둥이 프롬프트 쌍 만들기(CPU). 사용법: sbatch slurm/run_twin.sbatch <쌍 접두사> <출력 접두사>
#SBATCH -J dmp-twin
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
python3 -I scripts/30_make_twin_pairs.py "$1" "$2" | tee logs/twin_$2.txt
back logs/twin_$2.txt
back data/$2_meta.jsonl
