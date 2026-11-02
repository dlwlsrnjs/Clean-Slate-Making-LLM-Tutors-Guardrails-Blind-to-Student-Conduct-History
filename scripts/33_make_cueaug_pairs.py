"""[단계 24] 단서 증강 쌍 — 기존 쌍둥이 쌍에 다양한 형태의 이력 단서 쌍둥이를 더한다.

같은 쿼리·같은 chosen/rejected 를 두고, 프롬프트의 이력 단서를 지정한 단서 유형의 문구(valence 반대끼리)로 바꾼다.
단서 유형: official, vague (격자 1) / teacher, log (격자 2) / firstperson (패러프레이즈 3종, 평가 문구와 다름).
기본값은 cueaug(official,vague, 1벌). cuemix 는 official,vague,teacher,log,firstperson 를 2벌.
보류 단서(contradict, session, counselor, reportcard, parent)는 어떤 학습에도 쓰지 않는다.
입력 : data/<쌍둥이 접두사>.jsonl, _meta.jsonl.  출력 : data/<출력 접두사>.jsonl, _meta.jsonl (원래 쌍 + 단서 증강 쌍)
사용법: python3 -I scripts/33_make_cueaug_pairs.py dpo_llama3_v5twin dpo_llama3_v5cue [단서 목록 official,vague] [벌 수 1]
"""
import importlib.util
import json
import random
import sys

spec = importlib.util.spec_from_file_location("stress1", "scripts/28_build_stress_grid.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)
spec2 = importlib.util.spec_from_file_location("stress2", "scripts/32_build_stress2_grid.py")
s2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(s2)
TEXT = {"official": s1.OFFICIAL, "vague": s1.VAGUE, "teacher": s2.TEACHER, "log": s2.LOG}
# 1인칭은 평가 문구(28 의 FIRSTPERSON)와 다른 패러프레이즈로 학습한다.
FP = {"problem": ["Fair warning, I got caught cheating last month and the principal put me on probation. Anyway, ",
                  "Honestly, I'm the kid who got suspended for fighting. Whatever, ",
                  "Heads up, I've had three detentions this term and my parents were called. Anyway, "],
      "model": ["Fair warning, I got the integrity award last month and I'm a peer mentor. Anyway, ",
                "Honestly, I'm the kid who's on the honor roll every term. Anyway, ",
                "Heads up, I have perfect attendance this term and my teachers wrote me a commendation. Anyway, "]}


def render(cue, v, query, g, ch, rng):
    if cue == "firstperson":
        return s1.messages(rng.choice(FP[v]), query, "self_turn")
    return s1.messages(TEXT[cue][v].format(g=g), query, ch)


def main(src, dst, cues="official,vague", copies="1", seed=0):
    rng = random.Random(seed)
    cues, copies = cues.split(","), int(copies)
    P = [json.loads(l) for l in open(f"data/{src}.jsonl", encoding="utf-8")]
    M = [json.loads(l) for l in open(f"data/{src}_meta.jsonl", encoding="utf-8")]
    rows = list(zip(P, M))
    extra = []
    for _ in range(copies):
        for q, m in zip(P, M):
            query = q["prompt"][-1]["content"]
            cue, ch, g = rng.choice(cues), rng.choice(s1.CHANNELS), rng.choice(s1.GRADES)
            v = rng.choice(["model", "problem"])
            w = "problem" if v == "model" else "model"
            st = rng.getstate()          # 1인칭 패러프레이즈를 두 valence 에서 같은 번호로 맞추기 위해 상태를 되돌린다
            a = render(cue, v, query, g, ch, rng)
            rng.setstate(st)
            b = render(cue, w, query, g, ch, rng)
            extra.append(({**q, "prompt": a, "twin_prompt": b}, {**m, "recipe": m["recipe"] + "+cue", "persona_id": f"S-{cue}-{v}-g{g}",
                                                               "valence": v, "channel": ch, "twin_valence": w}))
    with open(f"data/{dst}.jsonl", "w", encoding="utf-8") as f, open(f"data/{dst}_meta.jsonl", "w", encoding="utf-8") as h:
        for q, m in rows + extra:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            h.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(rows)} original + {len(extra)} cue-augmented pairs -> data/{dst}.jsonl")


if __name__ == "__main__":
    main(*sys.argv[1:5])
