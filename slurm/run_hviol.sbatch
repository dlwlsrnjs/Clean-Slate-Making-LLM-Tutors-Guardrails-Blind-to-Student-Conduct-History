#!/bin/bash
# [단계 6-2] H 합성 위반 응답 생성(base) → 32B 판정 → 2라운드 채굴(원본 train 판정 + dpo3 on-policy + H 위반)까지 한 번에.
# 사용법: sbatch slurm/run_hviol.sbatch <HF 모델 id> <출력 접두사 예: dpo_llama3_v4>
#SBATCH -J dmp-hviol
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
fetch data/labels_v2.json
.venv-gen/bin/python scripts/21_gen_h_violations.py --model "$1" --gens data/gens_llama3_train.jsonl --out data/gens_llama3_hviol.jsonl -n 2
.venv-gen/bin/python scripts/12_judge.py --gens data/gens_llama3_hviol.jsonl --out data/judged_llama3_hviol_q32.jsonl --judge Qwen/Qwen3-32B --rubric v2 --labels data/labels_v2.json
python3 -I - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/judged_llama3_hviol_q32.jsonl", encoding="utf-8")]
print("H 합성 위반 판정:", collections.Counter(r["decision"] for r in rows), "| 쿼리별 comply:", collections.Counter(r["qid"] for r in rows if r["decision"] == "comply"))
PY
: > data/judged_$2_merged.jsonl
for t in llama3_train_q32 llama3_dpo3_train_q32 llama3_hviol_q32; do cat data/judged_$t.jsonl >> data/judged_$2_merged.jsonl; done
python3 -I scripts/03b_mine_dpo_pairs_v3.py data/judged_$2_merged.jsonl data/$2 | tee logs/mine_$2.txt
back logs/mine_$2.txt
back data/$2_meta.jsonl
