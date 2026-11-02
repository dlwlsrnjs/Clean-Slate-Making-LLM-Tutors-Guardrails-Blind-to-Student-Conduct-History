#!/bin/bash
# [단계 19] 스트레스 프로빙: 격자(최초 1회 생성) → 응답 생성 → 32B 판정 → PSG 진단. 모델 하나당 job 하나.
# 사용법: sbatch slurm/run_stress.sbatch <HF 모델 id> <태그> [LoRA 디렉터리]
# 예)   sbatch slurm/run_stress.sbatch NousResearch/Meta-Llama-3-8B-Instruct llama3_stress
#       sbatch slurm/run_stress.sbatch NousResearch/Meta-Llama-3-8B-Instruct llama3_dpo5_stress runs/dpo5
#SBATCH -J dmp-stress
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
export HF_HOME=/resilience/ljk98/hf_cache
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
cp /home/ljk98/dm-persona/data/labels_v2.json data/labels_v2.json
MODEL=$1; TAG=$2; LORA=${3:-}
# 격자는 모든 모델이 공유한다(base 의 중립 거부율로 쿼리를 골랐기 때문). LoRA 가 없는 base job 이 항상 새로 만들고, LoRA job 은 그것을 그대로 쓴다.
if [ -z "$LORA" ]; then python3 -I scripts/28_build_stress_grid.py 0.15 0.85 | tee logs/stress_grid.txt; fi
cp -f logs/stress_grid.txt /home/ljk98/dm-persona/logs/ 2>/dev/null || true
.venv-gen/bin/python scripts/11_generate.py --model "$MODEL" --grid data/grid_stress.jsonl --out data/gens_$TAG.jsonl -n 3 ${LORA:+--lora "$LORA"}
# 32B 판정자(v2.1 기준) + 재정의 gold 로 판정. 출력 이름을 judged_<태그>_q32.jsonl 로 맞춰 25_psg_diagnostics.py 가 바로 읽게 한다.
.venv-gen/bin/python scripts/12_judge.py --gens data/gens_$TAG.jsonl --out data/judged_${TAG}_q32.jsonl \
    --judge Qwen/Qwen3-32B --rubric v2 --labels data/labels_v2.json
