#!/bin/bash
# [단계 24] 단서 증강 쌍 만들기(CPU). 사용법: sbatch slurm/run_cueaug.sbatch <쌍 접두사> <출력 접두사> [단서 목록] [벌 수]
#SBATCH -J dmp-cueaug
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
python3 -I scripts/33_make_cueaug_pairs.py "$1" "$2" "${3:-official,vague}" "${4:-1}" | tee logs/cueaug_$2.txt
back logs/cueaug_$2.txt
back data/$2_meta.jsonl
