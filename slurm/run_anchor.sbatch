#!/bin/bash
# [단계 6-3~6-5] 거절 앵커 구축: 쿼리 증강(32B) → 격자 → 위반/좋은 거절 응답 생성(base) → 32B 판정 → 앵커 쌍 채굴 → 기존 쌍과 병합.
# REUSE=1 이면 증강 쿼리·격자·위반 응답은 재사용하고 좋은 거절 응답만 다시 만든다(REF_MODEL 로 작성 모델 지정, 예: Qwen/Qwen3-32B).
# 사용법: [REUSE=1] sbatch --export=ALL slurm/run_anchor.sbatch <HF base 모델 id> <기존 쌍 접두사 예: dpo_llama3_v4> <출력 접두사 예: dpo_llama3_v5>
#SBATCH -J dmp-anchor
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
fetch data/labels_v2.json
PY=.venv-gen/bin/python
if [ "${MINE_ONLY:-0}" != "1" ]; then
if [ "${REUSE:-0}" = "1" ]; then
    $PY scripts/23_gen_anchor_responses.py --model "${REF_MODEL:-$1}" --grid data/grid_aug.jsonl --per-query 6 --only ref --n-ref 3
    KINDS="ref"
  else
    $PY scripts/22_augment_refusal_queries.py data/queries.jsonl data/labels_v2.json
    python3 -I scripts/02_build_probe_grid.py data/queries_aug.jsonl data/personas.jsonl data/grid_aug.jsonl
    $PY scripts/23_gen_anchor_responses.py --model "$1" --grid data/grid_aug.jsonl --per-query 6
    KINDS="viol ref"
  fi
  for k in $KINDS; do
    $PY scripts/12_judge.py --gens data/gens_aug_$k.jsonl --out data/judged_aug_${k}_q32.jsonl --queries data/queries_aug.jsonl \
        --judge Qwen/Qwen3-32B --rubric v2 --labels data/labels_aug.json
  done
fi
python3 -I - <<'PY'
import json, collections
for k in ("viol", "ref"):
    rows = [json.loads(l) for l in open(f"data/judged_aug_{k}_q32.jsonl", encoding="utf-8")]
    print(k, collections.Counter((r["stratum"], r["decision"]) for r in rows))
PY
cat data/judged_aug_viol_q32.jsonl data/judged_aug_ref_q32.jsonl > data/judged_aug_merged.jsonl
python3 -I scripts/03b_mine_dpo_pairs_v3.py data/judged_aug_merged.jsonl data/dpo_aug_anchor --no-balance --max-pairs-per-query 3 | tee logs/mine_dpo_aug_anchor.txt
# 병합: 앵커는 H_CAP(기본 250) + AMH_CAP(기본 150)쌍만 쓰고 기존 쌍과 섞는다. MINE_ONLY=1 이면 생성·판정은 건너뛰고 채굴·병합만 다시 한다.
python3 -I - "$2" "$3" <<'PY'
import json, random, sys
old, out = sys.argv[1], sys.argv[2]
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
P0, M0 = rd(f"data/{old}.jsonl"), rd(f"data/{old}_meta.jsonl")
PA, MA = rd("data/dpo_aug_anchor.jsonl"), rd("data/dpo_aug_anchor_meta.jsonl")
rng = random.Random(0)
idx = list(range(len(PA))); rng.shuffle(idx)
import os
CAP = {"H": int(os.environ.get("H_CAP", 250)), "AM-H": int(os.environ.get("AMH_CAP", 150))}  # 앵커 상한(층별)
keep, cnt = [], {"H": 0, "AM-H": 0}
for i in idx:
    st = MA[i]["stratum"]
    if cnt[st] < CAP.get(st, 0):
        cnt[st] += 1; keep.append(i)
for i in keep:
    MA[i]["recipe"] = "ANCHOR"
rows = [(p, m) for p, m in zip(P0, M0)] + [(PA[i], MA[i]) for i in keep]
rng.shuffle(rows)
with open(f"data/{out}.jsonl", "w", encoding="utf-8") as f, open(f"data/{out}_meta.jsonl", "w", encoding="utf-8") as g:
    for p, m in rows:
        f.write(json.dumps(p, ensure_ascii=False) + "\n"); g.write(json.dumps(m, ensure_ascii=False) + "\n")
import collections
print(f"merged {len(P0)} old + {len(keep)} anchor = {len(rows)}; anchor by stratum {cnt}; by recipe", dict(collections.Counter(m['recipe'] for _, m in rows)))
print("by stratum", dict(collections.Counter(m['stratum'] for _, m in rows)))
PY
back logs/mine_dpo_aug_anchor.txt
back data/$3_meta.jsonl data/dpo_aug_anchor_meta.jsonl
