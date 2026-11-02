"""[단계 15-2] H(운영형 유해 요청)에서 comply 가 나온 쿼리를 모델별로 센다 — 안전 회귀가 특정 쿼리에 몰렸는지 확인.
사용법: python3 -I scripts/20_h_comply_by_query.py <tag> [<tag> ...]   (data/judged_<tag>_q32.jsonl 사용)"""
import json
import sys
from collections import Counter

for t in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"data/judged_{t}_q32.jsonl", encoding="utf-8")]
    h = [r for r in rows if r["stratum"] == "H"]
    c = Counter(r["qid"] for r in h if r["decision"] == "comply")
    n = Counter(r["qid"] for r in h)
    print(f"{t}: H n={len(h)} comply={sum(c.values())} ({sum(c.values()) / len(h):.3f}) queries with comply={len(c)}/{len(n)}")
    for q, k in c.most_common(6):
        print(f"   {q}  {k}/{n[q]}")
