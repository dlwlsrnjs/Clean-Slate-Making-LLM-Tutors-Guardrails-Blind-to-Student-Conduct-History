"""[단계 6] 판정이 끝난 생성 결과에서 DPO 선호쌍(prompt, chosen, rejected)을 채굴한다.

입력 : data/judged_*.jsonl — 생성 1건당 1줄. 필요한 필드:
         qid, stratum, gold(decline|comply|none), q_split, persona_id, valence, family, intensity, channel,
         sample_idx, messages, response, decision(comply|safe_complete|decline),
         persona_ref(응답이 학생 이력을 언급했는지), quality(높을수록 좋은 응답)
출력 : <out>.jsonl       — TRL 대화형 형식 {"prompt":[...], "chosen":[...], "rejected":[...]}
       <out>_meta.jsonl  — 같은 순서의 메타데이터(레시피, 층, 페르소나 등). 학습에는 쓰지 않고 분석용.

세 가지 레시피
  R1 공유-chosen : 쿼리마다 '정답 행동을 한 최고 품질 응답' y* 하나를 고르고, 그 쿼리에서 정답을 어긴 프롬프트마다
                   (prompt, chosen=y*, rejected=실제로 나온 위반 응답) 쌍을 만든다. 페르소나가 달라도 chosen 이 같아서
                   '이력과 무관하게 같은 행동'이 학습 목표가 된다.
  R2 교차 이식   : 같은 (쿼리, 표현군, 강도, 채널, 샘플)에서 문제/모범 페르소나의 결정이 갈릴 때, 정답 쪽 응답을
                   틀린 쪽 프롬프트의 chosen 으로 옮긴다. 편향이 실제로 나타난 지점만 쓰는 어려운 사례 집합.
  R3 균형        : 'decline 이 정답인 쌍'과 'comply 가 정답인 쌍'을 1:1로 맞춘다(한쪽으로만 치우쳐 학습되는 것 방지).
사용법: python3 -I scripts/03_mine_dpo_pairs.py data/judged.jsonl data/dpo_pairs [--max-rejected-per-prompt 2] [--hedge-reject]
"""
import json
import random
import sys
from collections import defaultdict


def is_correct(gold, decision):
    """정답 행동인가. decline 이 정답이면 거절/안전한 부분수행 모두 인정, comply 가 정답이면 수행만 인정."""
    return decision in ("decline", "safe_complete") if gold == "decline" else decision == "comply"


HEDGE_REJECT = False  # --hedge-reject: comply 가 정답인 쿼리에서 safe_complete(부분 수행+우회)도 위반으로 본다


def is_violation(gold, decision):
    """명백한 위반인가. 기본은 safe_complete 를 comply 정답 쪽에서 '위반도 정답도 아님'으로 두고 쌍에 쓰지 않는다.
    HEDGE_REJECT 이면 comply 정답 쿼리에서 safe_complete 도 위반이다(정상 요청에 대한 과잉 우회를 직접 벌주기 위함)."""
    if gold == "decline":
        return decision == "comply"
    return decision == "decline" or (HEDGE_REJECT and decision == "safe_complete")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def pair(prompt_row, chosen_row, rejected_row, recipe):
    """(TRL 형식 쌍, 메타데이터) 튜플을 만든다. 프롬프트는 prompt_row 의 것을 쓴다."""
    return ({"prompt": prompt_row["messages"],
             "chosen": [{"role": "assistant", "content": chosen_row["response"]}],
             "rejected": [{"role": "assistant", "content": rejected_row["response"]}]},
            {"recipe": recipe, "qid": prompt_row["qid"], "stratum": prompt_row["stratum"], "gold": prompt_row["gold"],
             "persona_id": prompt_row["persona_id"], "valence": prompt_row["valence"],
             "channel": prompt_row["channel"]})


def main(in_path, out_prefix, max_rej=2, seed=0):
    rng = random.Random(seed)
    # 학습용 쿼리 + 정답이 있는 쿼리 + 학습용 페르소나 표현군(또는 이력 없음)만 사용 → 시험 데이터 누수 방지
    rows = [r for r in load(in_path) if r["q_split"] == "train" and r["gold"] in ("decline", "comply")]
    rows = [r for r in rows if r.get("p_split", "train") in ("train", "all")]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)

    pairs = []
    # ---- R1: 쿼리별 공유 chosen y* ----
    for qid, rs in by_q.items():
        # y* 후보: 정답 행동이고 학생 이력을 언급하지 않은 응답(이력을 언급한 응답은 다른 페르소나 프롬프트에 옮기면 어색함)
        good = [r for r in rs if is_correct(r["gold"], r["decision"]) and not r["persona_ref"]]
        if not good:
            continue  # 정답 응답이 하나도 없으면 이 쿼리는 쌍을 만들지 않는다(억지로 만들지 않음)
        ystar = max(good, key=lambda r: (r["quality"], -len(r["response"])))  # 품질 최고, 동점이면 더 짧은 응답
        per_prompt = defaultdict(list)
        for r in rs:
            if is_violation(r["gold"], r["decision"]):
                per_prompt[(r["persona_id"], r["channel"])].append(r)
        for viols in per_prompt.values():
            for v in rng.sample(viols, min(max_rej, len(viols))):  # 프롬프트당 위반 응답 최대 max_rej개
                pairs.append(pair(v, ystar, v, "R1"))
    # ---- R2: 문제 vs 모범 페르소나 결정이 갈린 곳에서 정답 응답을 교차 이식 ----
    groups = defaultdict(dict)
    for r in rows:
        if r["valence"] in ("problem", "model"):
            groups[(r["qid"], r["family"], r["intensity"], r["channel"], r["sample_idx"])].setdefault(r["valence"], r)
    for g in groups.values():
        if "problem" in g and "model" in g:
            a, b = g["problem"], g["model"]
            for bad, good in ((a, b), (b, a)):  # 양방향: 문제가 틀리고 모범이 맞은 경우, 그 반대 경우
                if is_violation(bad["gold"], bad["decision"]) and is_correct(good["gold"], good["decision"]) \
                        and not good["persona_ref"]:
                    pairs.append(pair(bad, good, bad, "R2"))
    # ---- R3: decline-정답 쌍과 comply-정답 쌍을 같은 수로 맞춤 ----
    decline_type = [p for p in pairs if p[1]["gold"] == "decline"]   # p = (쌍, 메타) 이므로 p[1]이 메타
    comply_type = [p for p in pairs if p[1]["gold"] == "comply"]
    k = min(len(decline_type), len(comply_type))
    kept = rng.sample(decline_type, k) + rng.sample(comply_type, k)
    rng.shuffle(kept)

    with open(out_prefix + ".jsonl", "w", encoding="utf-8") as f, \
            open(out_prefix + "_meta.jsonl", "w", encoding="utf-8") as m:
        for p, meta in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            m.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"mined {len(pairs)} (decline-type {len(decline_type)}, comply-type {len(comply_type)}); kept {len(kept)}")
    for rec in ("R1", "R2"):
        print(f"  {rec}: {sum(1 for _, m_ in kept if m_['recipe'] == rec)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    mr = int(args[args.index("--max-rejected-per-prompt") + 1]) if "--max-rejected-per-prompt" in args else 2
    HEDGE_REJECT = "--hedge-reject" in args
    main(args[0], args[1], max_rej=mr)
