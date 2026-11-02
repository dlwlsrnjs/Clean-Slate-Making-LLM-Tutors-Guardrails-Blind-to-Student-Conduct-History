"""[단계 17] 대조 DPO(D2)용 쌍: 기존 쌍의 prompt 에서 페르소나 정보를 모두 지우고(= 같은 쿼리의 '페르소나 없음' 프롬프트로 교체)
chosen/rejected 는 그대로 둔다.

왜: dpo3~5 에서 PSG≈0 이 '페르소나를 보고 같은 판단을 하도록' 학습돼서인지, 쌍의 응답 내용(과거부 줄이기·H 거절)만으로 생긴 부수 효과인지
    가르기 위한 대조군. 학습 중 페르소나를 한 번도 보지 않는다. 쌍 수·응답·순서·하이퍼파라미터는 원본과 동일하다.
입력 : data/<원본 접두사>.jsonl(+_meta.jsonl), data/grid_llama3_train.jsonl, data/grid_aug.jsonl (qid 별 valence=none 행의 messages)
출력 : data/<출력 접두사>.jsonl(+_meta.jsonl)  — meta 에 control=True, prompt_orig_valence 를 추가
사용법: python3 -I scripts/26_make_control_pairs.py dpo_llama3_v5 dpo_llama3_v5ctrl
"""
import json
import sys
from collections import Counter

src, out = sys.argv[1:3]
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
P, M = rd(f"data/{src}.jsonl"), rd(f"data/{src}_meta.jsonl")
none_prompt = {}
for g in ("data/grid_llama3_train.jsonl", "data/grid_aug.jsonl"):
    for r in rd(g):
        if r["valence"] == "none":
            none_prompt.setdefault(r["qid"], r["messages"])
miss, mismatch, rows = 0, 0, []
for p, m in zip(P, M):
    base = none_prompt.get(m["qid"])
    if base is None:
        miss += 1
        continue
    if p["prompt"][-1]["content"] != base[-1]["content"]:   # 마지막 user 턴(=쿼리 본문)이 같아야 한다
        mismatch += 1
        continue
    q = dict(p)
    q["prompt"] = base
    rows.append((q, {**m, "control": True, "prompt_orig_valence": m["valence"], "valence": "none", "channel": "system_card"}))
dup = len(rows) - len({json.dumps(q, sort_keys=True, ensure_ascii=False) for q, _ in rows})
with open(f"data/{out}.jsonl", "w", encoding="utf-8") as f, open(f"data/{out}_meta.jsonl", "w", encoding="utf-8") as g:
    for q, m in rows:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
        g.write(json.dumps(m, ensure_ascii=False) + "\n")
print(f"original {len(P)} -> control {len(rows)}  (qid 없음 {miss}, 쿼리 본문 불일치 {mismatch}, 완전 중복 {dup})")
print("원래 프롬프트의 valence 분포:", dict(Counter(m["prompt_orig_valence"] for _, m in rows)))
print("고유 프롬프트 수:", len({json.dumps(q["prompt"], ensure_ascii=False) for q, _ in rows}), "| 고유 쿼리 수:", len({m["qid"] for _, m in rows}))
n_sys = sum(1 for q, _ in rows if len(q["prompt"]) == 2 and q["prompt"][0]["role"] == "system")
print(f"프롬프트가 [system, user] 2턴인 쌍: {n_sys}/{len(rows)}")
q, m = rows[0]
print("예:", json.dumps(q["prompt"], ensure_ascii=False)[:400])
