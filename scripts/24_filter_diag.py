"""[진단] 03b 의 clean() 조건별로 탈락하는 비율을 센다. 사용법: python3 -I scripts/24_filter_diag.py data/judged_x.jsonl [stratum]"""
import importlib.util
import json
import sys
from collections import Counter

spec = importlib.util.spec_from_file_location("m", "scripts/03b_mine_dpo_pairs_v3.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
if len(sys.argv) > 2:
    rows = [r for r in rows if r["stratum"] == sys.argv[2]]
c = Counter()
for r in rows:
    if not m.is_correct(r["gold"], r["decision"]):
        c["wrong_decision"] += 1
        continue
    c["correct"] += 1
    L = len(r["response"])
    h = r["stratum"] == "H"
    if r["persona_ref"]: c["persona_ref"] += 1
    if m.LEAK.search(r["response"]): c["leak"] += 1
    if r.get("likely_fabricated"): c["fabricated"] += 1
    if L > m.MAX_LEN: c["too_long"] += 1
    if m.ENDORSE.search(r["response"][:120]): c["endorse"] += 1
    if r["quality"] < (2 if h else 4): c["low_quality"] += 1
    if r["moralizing"] > (4 if h else m.MAX_MOR[r["gold"]]): c["moralizing"] += 1
    if L < (300 if h else 150): c["too_short"] += 1
    if m.clean(r): c["CLEAN"] += 1
print(len(rows), dict(c))
import statistics
print("len median", statistics.median(len(r["response"]) for r in rows), "quality mean", sum(r["quality"] for r in rows) / len(rows), "mor mean", sum(r["moralizing"] for r in rows) / len(rows))
