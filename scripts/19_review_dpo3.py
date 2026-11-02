"""[단계 15] DPO v3 직접 검토용 덤프 + 길이/거절문구/이력 누출 통계 (표준 라이브러리만 사용).

입력 : data/judged_<tag>_q32.jsonl (32B v2.1 재판정 결과) — base 와 비교할 태그들
출력 : 터미널 통계표, <out_md> 에 층별 무작위 샘플(요청 + 응답 + 판정 이유) 마크다운
통계 : 응답 평균 길이, 'I can't help'류 거절 문구 비율, 학년·이력 누출(정규식) 비율, 층별 decision 분포
사용법: python3 -I scripts/19_review_dpo3.py <out_md> <base_tag> <new_tag> [<tag> ...]
"""
import json
import random
import re
import sys
from collections import defaultdict

REFUSE = re.compile(r"\b(I can('|’)?t|I cannot|I('|’)?m (sorry|unable)|I won('|’)?t be able|can('|’)?t help with that)\b", re.I)
LEAK = re.compile(r"\b(grade|year)\s*\d+\b|\b\d+(st|nd|rd|th)[- ]grader\b|honou?r roll|\b(your|the) (record|history|file|conduct)\b|"
                  r"on file|last time|earlier,? you|as we discussed", re.I)


def load(tag):
    return [json.loads(l) for l in open(f"data/judged_{tag}_q32.jsonl", encoding="utf-8")]


def stats(tag, rows):
    n = len(rows)
    ln = sum(len(r["response"]) for r in rows) / n
    ref = sum(1 for r in rows if REFUSE.search(r["response"][:200])) / n
    leak = sum(1 for r in rows if LEAK.search(r["response"])) / n
    print(f"{tag:14s} n={n:6d} 평균길이={ln:6.0f}  거절문구(앞 200자)={ref:.3f}  이력누출(정규식)={leak:.3f}  persona_ref={sum(r['persona_ref'] for r in rows) / n:.3f}")


def main():
    out, base, *tags = sys.argv[1:]
    data = {t: load(t) for t in [base] + tags}
    for t, rows in data.items():
        stats(t, rows)
    new = tags[-1]
    rows = data[new]
    rng = random.Random(0)
    by = defaultdict(list)
    for r in rows:
        by[(r["stratum"], r["decision"])].append(r)
    want = [("H", "comply", 8), ("H", "safe_complete", 4), ("H", "decline", 3), ("HS", "comply", 4), ("HS", "decline", 4),
            ("AM-H", "safe_complete", 6), ("AM-H", "comply", 4), ("AM-H", "decline", 3), ("AM-N", "comply", 2),
            ("BL", "comply", 3), ("AMB", "comply", 3), ("AMB", "decline", 3), ("C", "comply", 2)]
    with open(out, "w", encoding="utf-8") as f:
        for st, dec, k in want:
            pool = by.get((st, dec), [])
            f.write(f"\n# {st} / {dec}  (후보 {len(pool)}건 중 {min(k, len(pool))}건)\n")
            for r in rng.sample(pool, min(k, len(pool))):
                f.write(f"\n## qid={r['qid']} {r['valence']} quality={r['quality']} mor={r['moralizing']} fab={r.get('likely_fabricated')}\n")
                f.write(f"**요청**: {r['messages'][-1]['content'][:600]}\n\n**응답**: {r['response'][:1400]}\n\n[reason] {r.get('reason', '')}\n")


if __name__ == "__main__":
    main()
