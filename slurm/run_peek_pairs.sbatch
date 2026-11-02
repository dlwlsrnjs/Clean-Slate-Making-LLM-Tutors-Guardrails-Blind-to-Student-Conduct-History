#!/bin/bash
# 채굴된 쌍 샘플을 사람이 읽을 수 있게 덤프한다. CPU job. 사용법: sbatch slurm/run_peek_pairs.sbatch <쌍 접두사> <층,층,...>
#SBATCH -J dmp-peek
#SBATCH -p all
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
cd /resilience/ljk98/dm-persona
python3 -I - "$1" "$2" <<'PY' > /home/ljk98/dm-persona/logs/review/peek_$1.md
import json, random, sys
pre, strata = sys.argv[1], set(sys.argv[2].split(","))
P = [json.loads(l) for l in open(f"data/{pre}.jsonl", encoding="utf-8")]
M = [json.loads(l) for l in open(f"data/{pre}_meta.jsonl", encoding="utf-8")]
rng = random.Random(1)
for st in sorted(strata):
    idx = [i for i, m in enumerate(M) if m["stratum"] == st]
    for i in rng.sample(idx, min(4, len(idx))):
        p, m = P[i], M[i]
        print(f"\n## {st} {m['recipe']} {m['qid']} {m['valence']}\n**요청**: {p['prompt'][-1]['content'][:500]}\n\n**CHOSEN** ({m['chosen_len']}자): {p['chosen'][0]['content'][:900]}\n\n**REJECTED** ({m['rejected_len']}자): {p['rejected'][0]['content'][:700]}\n")
PY
