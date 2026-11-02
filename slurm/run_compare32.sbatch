#!/bin/bash
# [단계 14] 32B 재판정 결과(judged_<tag>_q32.jsonl)로 PSG 표(05)와 base 대비 비교(06)를 낸다. CPU job.
# 사용법: sbatch slurm/run_compare32.sbatch "<base 태그>" "<비교할 태그 목록>"   예: "llama3" "llama3_dpo1 llama3_dpo2 llama3_dpo3"
#SBATCH -J dmp-cmp32
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
BASE=$1
OUT=logs/compare32_$(echo "$2" | tr ' ' '_').txt
{
for t in $BASE $2; do
  echo "######## PSG ($t, 32B v2.1, 거부=decline+safe_complete)"
  python3 -I scripts/05_analyze.py data/judged_${t}_q32.jsonl --safe-complete-as-refusal
done
for t in $2; do
  echo; echo "######## $BASE vs $t — test 쿼리 x test 표현군"
  python3 -I scripts/06_compare.py --run base=data/judged_${BASE}_q32.jsonl --run new=data/judged_${t}_q32.jsonl --q-split test --p-split test
  echo; echo "######## $BASE vs $t — 전체 dev+test"
  python3 -I scripts/06_compare.py --run base=data/judged_${BASE}_q32.jsonl --run new=data/judged_${t}_q32.jsonl
done
} | tee "$OUT"
back "$OUT"
