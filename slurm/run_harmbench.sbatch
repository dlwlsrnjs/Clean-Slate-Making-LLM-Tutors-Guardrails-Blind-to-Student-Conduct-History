#!/bin/bash
# [단계 10] 판정자와 독립인 안전성 점검(HarmBench 분류기). 사용법: sbatch slurm/run_harmbench.sbatch <기준 태그> <비교 태그...>
#SBATCH -J dmp-harmbench
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
# 분류기 가중치(13B, 약 26GB)가 resilience 캐시에 없으면 /home 캐시에서 복사한다
M=models--cais--HarmBench-Llama-2-13b-cls
[ -d $HF_HOME/hub/$M ] || { [ -d "$HOME/.cache/huggingface/hub/$M" ] && mkdir -p $HF_HOME/hub && rsync -a "$HOME/.cache/huggingface/hub/$M" $HF_HOME/hub/; } || true
args=()
for t in "$@"; do args+=(--gens "$t=data/gens_$t.jsonl"); done
.venv-gen/bin/python scripts/13_harmbench_check.py "${args[@]}" | tee logs/harmbench_check.txt
back logs/harmbench_check.txt
