#!/bin/bash
# [판정자 교차검증] 사람 라벨이 없는 동안, 검증용 200개를 다른 판정자(Llama-3, Mistral)로도 판정해 Qwen3 판정자와의 일치율을 본다.
# 판정자끼리 크게 갈리는 곳(특히 safe_complete 경계)이 판정이 불안정한 곳이다. 사람 라벨을 대체하지는 못한다.
#SBATCH -J dmp-xcheck
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
# 검증 CSV 를 12_judge.py 입력 형식(qid, response)으로 바꾼다
python3 -I - <<'PY'
import csv, json
key = {json.loads(l)["id"]: json.loads(l) for l in open("data/judge_validation_key.jsonl")}
with open("data/gens_valset.jsonl", "w") as f:
    for r in csv.DictReader(open("data/judge_validation_200.csv", encoding="utf-8-sig")):
        f.write(json.dumps({"id": int(r["id"]), "qid": key[int(r["id"])]["qid"], "response": r["reply"]}) + "\n")
PY
for spec in "NousResearch/Meta-Llama-3-8B-Instruct llama3" "mistralai/Mistral-7B-Instruct-v0.3 mistral"; do
  set -- $spec
  .venv-gen/bin/python scripts/12_judge.py --gens data/gens_valset.jsonl --out data/judged_valset_$2.jsonl --judge $1
done
python3 -I - <<'PY' | tee logs/xcheck_judges.txt
import json
from collections import Counter
key = {json.loads(l)["id"]: json.loads(l) for l in open("data/judge_validation_key.jsonl")}
C = ("comply", "safe_complete", "decline")
for tag in ("llama3", "mistral"):
    alt = {json.loads(l)["id"]: json.loads(l)["decision"] for l in open(f"data/judged_valset_{tag}.jsonl")}
    ids = sorted(alt)
    cm = Counter((key[i]["judge_decision"], alt[i]) for i in ids)
    agree = sum(cm[(c, c)] for c in C) / len(ids)
    binary = sum((key[i]["judge_decision"] != "comply") == (alt[i] != "comply") for i in ids) / len(ids)
    print(f"\n== Qwen3 판정자 vs {tag} 판정자 (n={len(ids)}): 3분류 일치 {agree:.3f}, 이분(거부=decline+safe_complete) 일치 {binary:.3f}")
    print("행=Qwen3, 열=" + tag)
    for a in C:
        print(f"{a:14s}" + "".join(f"{cm[(a, b)]:8d}" for b in C))
PY
back logs/xcheck_judges.txt
