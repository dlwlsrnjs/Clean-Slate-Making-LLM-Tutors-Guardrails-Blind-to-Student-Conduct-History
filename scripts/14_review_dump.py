"""[단계 11] 사람(연구자·Claude)이 직접 읽고 점검할 수 있도록 판정 결과·DPO 쌍·전/후 응답을 마크다운으로 덤프한다. (stdlib 만 사용)

만드는 파일(--out 디렉터리):
  review_judge.md : 판정자 결과 표본. 층별로 고르게, decision 별로 고르게 뽑아 요청/응답/판정을 나란히 본다.
  review_pairs.md : DPO 학습 쌍 표본. prompt(페르소나 카드 포함) / chosen / rejected 와 레시피·층·정답 행동.
  review_sbs.md   : 같은 프롬프트(쿼리·표현군·강도·채널·샘플)에서 '문제 vs 모범' 페르소나의 기준 응답과 DPO 후 응답을 나란히.
사용법: python3 -I scripts/14_review_dump.py data logs/review --base llama3 --new llama3_dpo2 --pairs dpo_llama3_v2
"""
import argparse
import json
import os
import random
from collections import defaultdict

STRATA = ("H", "AM-H", "BL", "AM-BL", "C", "AMB")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def clip(t, n):
    t = t.strip()
    return t if len(t) <= n else t[:n] + f" …[+{len(t) - n}자]"


def system_text(msgs):
    return next((m["content"] for m in msgs if m["role"] == "system"), "")


def last_user(msgs):
    return [m for m in msgs if m["role"] == "user"][-1]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    qtext = {q["qid"]: q["query"] for q in load(f"{a.data_dir}/queries.jsonl")}

    # ---- 1. 판정 표본: 층 x decision 칸에서 하나씩 돌려 뽑는다 (기준 모델, dev+test)
    base = [r for r in load(f"{a.data_dir}/judged_{a.base}.jsonl") if r["q_split"] in ("dev", "test")]
    cells = defaultdict(list)
    for r in base:
        cells[(r["stratum"], r["decision"])].append(r)
    for c in cells.values():
        rng.shuffle(c)
    picked, keys = [], sorted(cells)
    while len(picked) < 48 and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(picked) < 48:
                picked.append(cells[k].pop())
    with open(f"{a.out_dir}/review_judge.md", "w", encoding="utf-8") as f:
        f.write(f"# 판정 표본 ({a.base}, {len(picked)}개)\n\n판정자 결과는 [judge] 줄. 요청/응답은 판정자가 본 것과 같다.\n")
        for i, r in enumerate(picked, 1):
            f.write(f"\n---\n## J{i}  층={r['stratum']} 정답={r['gold']} 이력={r['valence']}\n"
                    f"**요청**: {qtext[r['qid']]}\n\n**응답**: {clip(r['response'], 1100)}\n\n"
                    f"[judge] decision={r['decision']} quality={r['quality']} moralizing={r['moralizing']} persona_ref={r['persona_ref']}\n")

    # ---- 2. DPO 쌍 표본
    pairs = load(f"{a.data_dir}/{a.pairs}.jsonl")
    meta = load(f"{a.data_dir}/{a.pairs}_meta.jsonl")
    idx = list(range(len(pairs)))
    by = defaultdict(list)
    for i in idx:
        by[(meta[i]["recipe"], meta[i]["gold"], meta[i]["stratum"])].append(i)
    sel, ks = [], sorted(by)
    for k in ks:
        rng.shuffle(by[k])
    while len(sel) < 40 and any(by[k] for k in ks):
        for k in ks:
            if by[k] and len(sel) < 40:
                sel.append(by[k].pop())
    with open(f"{a.out_dir}/review_pairs.md", "w", encoding="utf-8") as f:
        f.write(f"# DPO 쌍 표본 ({a.pairs}, {len(sel)}개 / 전체 {len(pairs)}개)\n\n"
                "chosen 은 '정답 행동을 한 응답', rejected 는 '그 프롬프트에서 실제로 나온 위반 응답'이다. "
                "chosen 이 다른 페르소나 프롬프트에서 온 경우도 있으니 prompt 의 페르소나와 chosen 의 내용이 어색하지 않은지 본다.\n")
        for n, i in enumerate(sel, 1):
            m, p = meta[i], pairs[i]
            f.write(f"\n---\n## P{n}  {m['recipe']} 층={m['stratum']} 정답={m['gold']} 프롬프트 이력={m['valence']} 채널={m['channel']}\n"
                    f"**시스템/페르소나**: {clip(system_text(p['prompt']), 500) or '(없음)'}\n\n"
                    f"**요청**: {last_user(p['prompt'])}\n\n"
                    f"**chosen**: {clip(p['chosen'][0]['content'], 900)}\n\n**rejected**: {clip(p['rejected'][0]['content'], 900)}\n")

    # ---- 3. 같은 프롬프트에서 문제 vs 모범, 기준 vs DPO 후
    new = {k: r for r in load(f"{a.data_dir}/judged_{a.new}.jsonl") for k in [(r["qid"], r["persona_id"], r["channel"], r["sample_idx"])]}
    g = defaultdict(dict)
    for r in base:
        if r["valence"] in ("problem", "model") and r["p_split"] == "test":
            g[(r["qid"], r["family"], r["intensity"], r["relevance"], r["channel"], r["sample_idx"])][r["valence"]] = r
    cand = [v for v in g.values() if "problem" in v and "model" in v and v["problem"]["decision"] != v["model"]["decision"]
            and v["problem"]["stratum"] in ("AM-H", "BL", "AM-BL", "C")]
    rng.shuffle(cand)
    cells = defaultdict(list)
    for v in cand:
        cells[v["problem"]["stratum"]].append(v)
    chosen = []
    while len(chosen) < 12 and any(cells.values()):
        for s in list(cells):
            if cells[s] and len(chosen) < 12:
                chosen.append(cells[s].pop())
    with open(f"{a.out_dir}/review_sbs.md", "w", encoding="utf-8") as f:
        f.write(f"# 전/후 나란히 ({a.base} → {a.new}), 기준에서 문제/모범 결정이 갈린 사례 {len(chosen)}개\n")
        for n, v in enumerate(chosen, 1):
            p, m = v["problem"], v["model"]
            f.write(f"\n---\n## S{n}  층={p['stratum']} 정답={p['gold']} 채널={p['channel']}\n**요청**: {qtext[p['qid']]}\n\n"
                    f"**문제 이력 페르소나**: {clip(system_text(p['messages']), 380) or '(이전 대화 턴으로 제공)'}\n\n"
                    f"**모범 이력 페르소나**: {clip(system_text(m['messages']), 380) or '(이전 대화 턴으로 제공)'}\n\n")
            for lab, r in (("기준/문제", p), ("기준/모범", m)):
                f.write(f"### {lab} [judge {r['decision']}]\n{clip(r['response'], 700)}\n\n")
            for lab, r in (("DPO후/문제", p), ("DPO후/모범", m)):
                x = new.get((r["qid"], r["persona_id"], r["channel"], r["sample_idx"]))
                if x:
                    f.write(f"### {lab} [judge {x['decision']}]\n{clip(x['response'], 700)}\n\n")
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
