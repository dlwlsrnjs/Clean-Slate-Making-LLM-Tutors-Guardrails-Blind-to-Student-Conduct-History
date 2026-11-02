#!/bin/bash
# [단계 17] 대조 DPO 쌍 만들기(CPU). 사용법: sbatch slurm/run_ctrlpairs.sbatch <원본 쌍 접두사> <출력 접두사>
#SBATCH -J dmp-ctrlpairs
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
python3 -I scripts/26_make_control_pairs.py "$1" "$2" | tee logs/ctrlpairs_$2.txt
back logs/ctrlpairs_$2.txt
back data/$2_meta.jsonl
