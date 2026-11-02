"""[단계 6, v3] 직접 검토(AUDIT_review.md)에서 드러난 쌍 품질 문제를 고친 선호쌍 채굴. 입력은 32B 판정자(v2 기준) + 재정의 gold 로 만든 judged 파일.

03 대비 바뀐 점
1. chosen 후보를 '깨끗한' 응답으로 제한한다: 이력/학년/기록 언급 없음, 환각 의심 없음(likely_fabricated), quality>=4,
   moralizing 이 낮음(comply 정답 <=2, decline 정답 <=3), 너무 짧은 한 줄 거절 제외(>=150자).
2. 길이 편향 제거: chosen 은 '가장 짧은 것'이 아니라 후보들의 중앙 길이에 가장 가까운 것(품질 동점 시). comply 정답 쌍은
   chosen 이 rejected 의 절반보다 짧으면 제외한다.
3. rejected 는 chosen 보다 quality 가 높지 않아야 한다(더 좋은 응답을 rejected 로 쓰지 않음).
4. 쿼리당 쌍 수 상한(--max-pairs-per-query, 기본 8) — 같은 쿼리가 학습을 지배하지 않게 한다.
5. 정답 위반 정의에 hedge(comply 정답에서 safe_complete)를 기본 포함한다. 단 HS·AM-N 은 400자 미만 단순 거절만 위반으로 본다.
레시피 R1(쿼리별 공유 chosen)·R2(문제↔모범 교차 이식)·R3(decline형:comply형 1:1)은 03 과 같다.
사용법: python3 -I scripts/03b_mine_dpo_pairs_v3.py data/judged_llama3_train_q32.jsonl data/dpo_llama3_v3 [--no-balance] [--max-pairs-per-query N]
"""
import json
import random
import re
import sys
from collections import defaultdict
from statistics import median

LEAK = re.compile(r"\b(grade|year)\s*\d+\b|\b\d+(st|nd|rd|th)[- ]grader\b|honou?r roll|\b(your|the) (record|history|file|conduct)\b|"
                  r"on file|last time|earlier,? you|as we discussed|hi again|welcome back|nice to see you again|you('ve| have) (been|had) (some|a|in)", re.I)
ENDORSE = re.compile(r"^\W*(i think )?(that('s| is| sounds)|this is|what) (like )?(a |an )?(really |very )?(great|good|wonderful|fantastic|awesome|fun) (idea|plan)|^\W*i('m| am) glad you('re| are) (thinking|planning)", re.I)
MAX_MOR = {"comply": 2, "decline": 3}
MAX_LEN = 2200  # chosen 상한: v3 1라운드에서 응답이 base 보다 29% 길어진 장문화를 막는다


def is_correct(gold, decision):
    return decision in ("decline", "safe_complete") if gold == "decline" else decision == "comply"


def is_violation(r):
    """rejected 로 쓸 만한 오답인가. HS·AM-N(원래 도와야 하는 안전·규범 질문)은 실질 없는 짧은 단순 거절만 오답으로 친다
    (길게 설명하고 대안을 준 응답까지 오답으로 치면 정당한 부분 응답을 벌주게 된다)."""
    if r["gold"] == "decline":
        return r["decision"] == "comply"
    if r["stratum"] in ("HS", "AM-N"):
        return r["decision"] == "decline" and len(r["response"]) < 400
    return r["decision"] in ("decline", "safe_complete")


def clean(r):
    """chosen 으로 써도 되는 응답인가. H 는 정상 거절이 훈계 점수가 높고 quality 가 낮게 나오므로(rejected 가 위반 응답이라 충분히
    선호할 만하다) 기준을 따로 둔다: quality>=2, moralizing<=4, 길이>=300(대안·설명이 있는 거절)."""
    base_ok = (is_correct(r["gold"], r["decision"]) and not r["persona_ref"] and not LEAK.search(r["response"])
               and not r.get("likely_fabricated") and len(r["response"]) <= MAX_LEN
               and not ENDORSE.search(r["response"][:120]))   # 위험한 계획에 '좋은 생각이에요'로 시작하는 응답은 제외
    if r["stratum"] == "H":
        return base_ok and r["quality"] >= 2 and r["moralizing"] <= 4 and len(r["response"]) >= 300
    return base_ok and r["quality"] >= 4 and r["moralizing"] <= MAX_MOR[r["gold"]] and len(r["response"]) >= 150


def pair(prompt_row, chosen_row, rejected_row, recipe):
    return ({"prompt": prompt_row["messages"],
             "chosen": [{"role": "assistant", "content": chosen_row["response"]}],
             "rejected": [{"role": "assistant", "content": rejected_row["response"]}]},
            {"recipe": recipe, "qid": prompt_row["qid"], "stratum": prompt_row["stratum"], "gold": prompt_row["gold"],
             "persona_id": prompt_row["persona_id"], "valence": prompt_row["valence"], "channel": prompt_row["channel"],
             "chosen_len": len(chosen_row["response"]), "rejected_len": len(rejected_row["response"])})


def acceptable(prompt_row, chosen_row, rejected_row):
    if rejected_row["quality"] > chosen_row["quality"]:
        return False
    if prompt_row["gold"] == "comply" and len(chosen_row["response"]) < 0.5 * len(rejected_row["response"]):
        return False
    return True


def main(in_path, out_prefix, max_rej=2, max_per_q=8, seed=0, balance=True):
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    rows = [r for r in rows if r["q_split"] == "train" and r["gold"] in ("decline", "comply") and r.get("p_split", "train") in ("train", "all")]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)

    pairs = []
    # ---- R1 ----
    for qid, rs in by_q.items():
        good = [r for r in rs if clean(r)]
        if not good:
            continue
        mid = median(len(r["response"]) for r in good)
        top_q = max(r["quality"] for r in good)
        cand = [r for r in good if r["quality"] == top_q]
        ystar = min(cand, key=lambda r: (abs(len(r["response"]) - mid), rng.random()))
        per_prompt = defaultdict(list)
        for r in rs:
            if is_violation(r):
                per_prompt[(r["persona_id"], r["channel"])].append(r)
        for viols in per_prompt.values():
            for v in rng.sample(viols, min(max_rej, len(viols))):
                if acceptable(v, ystar, v):
                    pairs.append(pair(v, ystar, v, "R1"))
    # ---- R2 ----
    groups = defaultdict(dict)
    for r in rows:
        if r["valence"] in ("problem", "model"):
            groups[(r["qid"], r["family"], r["intensity"], r["channel"], r["sample_idx"])].setdefault(r["valence"], r)
    for g in groups.values():
        if "problem" in g and "model" in g:
            a, b = g["problem"], g["model"]
            for bad, good in ((a, b), (b, a)):
                if is_violation(bad) and clean(good) and acceptable(bad, good, bad):
                    pairs.append(pair(bad, good, bad, "R2"))
    # ---- 쿼리당 상한 ----
    rng.shuffle(pairs)
    per_q, capped = defaultdict(int), []
    for p in pairs:
        if per_q[p[1]["qid"]] < max_per_q:
            per_q[p[1]["qid"]] += 1
            capped.append(p)
    # ---- R3 ----
    decline_type = [p for p in capped if p[1]["gold"] == "decline"]
    comply_type = [p for p in capped if p[1]["gold"] == "comply"]
    if balance:
        k = min(len(decline_type), len(comply_type))
        kept = rng.sample(decline_type, k) + rng.sample(comply_type, k)
    else:                                   # --no-balance: 거절 앵커 쌍(decline형만)을 따로 채굴할 때
        kept = list(capped)
    rng.shuffle(kept)
    with open(out_prefix + ".jsonl", "w", encoding="utf-8") as f, open(out_prefix + "_meta.jsonl", "w", encoding="utf-8") as m:
        for p, meta in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            m.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"queries with a clean chosen: {sum(1 for rs in by_q.values() if any(clean(r) for r in rs))}/{len(by_q)}")
    print(f"mined {len(pairs)} -> per-query cap {len(capped)} (decline-type {len(decline_type)}, comply-type {len(comply_type)}); kept {len(kept)}")
    for rec in ("R1", "R2"):
        print(f"  {rec}: {sum(1 for _, m_ in kept if m_['recipe'] == rec)}")
    bys = defaultdict(int)
    for _, m_ in kept:
        bys[m_["stratum"]] += 1
    print("  by stratum:", dict(sorted(bys.items())))
    print(f"  mean chosen/rejected length: {sum(m_['chosen_len'] for _, m_ in kept) / len(kept):.0f} / {sum(m_['rejected_len'] for _, m_ in kept) / len(kept):.0f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], max_rej=int(a[a.index("--max-rejected-per-prompt") + 1]) if "--max-rejected-per-prompt" in a else 2,
         max_per_q=int(a[a.index("--max-pairs-per-query") + 1]) if "--max-pairs-per-query" in a else 8,
         balance="--no-balance" not in a)
