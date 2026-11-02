#!/bin/bash
# [단계 8] 전/후 비교표. CPU 만 쓰는 짧은 job. 사용법: sbatch slurm/run_compare.sbatch <기준 태그> <비교 태그> <출력 이름>
#SBATCH -J dmp-compare
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
{ python3 -I scripts/06_compare.py --run base=data/judged_$1.jsonl --run new=data/judged_$2.jsonl --q-split test --p-split test
  echo; echo "######## 전체 dev+test, 모든 표현군"
  python3 -I scripts/06_compare.py --run base=data/judged_$1.jsonl --run new=data/judged_$2.jsonl; } | tee logs/compare_$3.txt
back logs/compare_$3.txt
