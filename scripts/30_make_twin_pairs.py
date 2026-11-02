"""[단계 21] 반사실 쌍둥이 프롬프트 만들기 — CPC-DPO(반사실 페르소나 일관 DPO)의 입력.

각 선호쌍 (프롬프트 x_p, chosen, rejected) 에 대해 같은 쿼리·같은 응답은 두고 x_p 안의 페르소나 문단만 '반대 valence' 로 바꾼 x_q 를 붙인다.
  model <-> problem 교체. neutral 은 model/problem 중 시드 고정 무작위. none(이력 없음)은 시스템 카드에 무작위 페르소나를 추가.
교체 대상은 같은 (표현군, 도메인, 강도, 학년) 의 페르소나라서 길이·형식이 같고 valence 만 달라진다.
입력 : data/<쌍 접두사>.jsonl, data/<쌍 접두사>_meta.jsonl, data/personas.jsonl
출력 : data/<출력 접두사>.jsonl (원래 필드 + twin_prompt), data/<출력 접두사>_meta.jsonl (+ twin_valence)
사용법: python3 -I scripts/30_make_twin_pairs.py dpo_llama3_v5 dpo_llama3_v5twin
"""
import json
import random
import sys
from collections import Counter

TUTOR_SYSTEM_PREFIX = "You are a friendly AI tutor for school students."


def rd(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def swap(messages, old, new):
    """messages 안에서 old 문단을 new 로 바꾼 사본. old 가 없으면 None."""
    out, hit = [], False
    for m in messages:
        c = m["content"]
        if old and old in c:
            c, hit = c.replace(old, new), True
        out.append({**m, "content": c})
    return out if hit else None


def main(src, dst, seed=0):
    rng = random.Random(seed)
    P, M = rd(f"data/{src}.jsonl"), rd(f"data/{src}_meta.jsonl")
    bank = {p["persona_id"]: p for p in rd("data/personas.jsonl")}
    by_cell = {(p["family"], p["domain"], p["intensity"], p["grade"], p["valence"]): p
               for p in bank.values() if p["valence"] != "none"}
    rows, miss = [], Counter()
    for q, m in zip(P, M):
        pid = m["persona_id"]
        if pid == "none":  # 이력 없음: 시스템 카드에 무작위 페르소나를 붙인 쌍둥이
            twin_p = bank[rng.choice([k for k in bank if k.startswith("A-") and ("-model-" in k or "-problem-" in k)])]
            msgs = [dict(x) for x in q["prompt"]]
            if msgs[0]["role"] == "system":
                msgs[0]["content"] += "\n\n" + twin_p["text"]
                tw = msgs
            else:
                tw = None
        else:
            p = bank[pid]
            tv = {"model": "problem", "problem": "model"}.get(p["valence"]) or rng.choice(["model", "problem"])
            twin_p = by_cell.get((p["family"], p["domain"], p["intensity"], p["grade"], tv))
            tw = swap(q["prompt"], p["text"], twin_p["text"]) if twin_p else None
        if tw is None:
            miss[m["valence"]] += 1
            continue
        rows.append(({**q, "twin_prompt": tw}, {**m, "twin_valence": twin_p["valence"]}))
    with open(f"data/{dst}.jsonl", "w", encoding="utf-8") as f, open(f"data/{dst}_meta.jsonl", "w", encoding="utf-8") as g:
        for q, m in rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            g.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(rows)}/{len(P)} pairs with twin; dropped by valence: {dict(miss)}")
    print("원래 valence -> 쌍둥이 valence:", dict(Counter((m['valence'], m['twin_valence']) for _, m in rows)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
