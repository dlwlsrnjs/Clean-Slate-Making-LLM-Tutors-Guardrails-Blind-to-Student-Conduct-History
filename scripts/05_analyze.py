"""[단계 5] 판정된 생성 결과로 '페르소나 민감도'를 분석한다 (표준 라이브러리만 사용).

입력 : data/judged_<tag>.jsonl (12_judge.py 출력)
출력 : 터미널 표 — 층(stratum)별로
         거부율(이력 없음 / 무관 이력 / 모범 / 문제), PSG, 95% 신뢰구간, flip 비율, strict/lenient 건수, McNemar p

용어
  PSG     = 문제 페르소나 거부율 - 모범 페르소나 거부율. 양수면 '문제아 이력일 때 더 많이 거부'.
  flip    = 같은 조건에서 문제/모범 페르소나의 거부 여부가 서로 다른 비율.
  strict  = 문제 페르소나만 거부한 건수(문제아에게만 더 엄격).
  lenient = 모범 페르소나만 거부한 건수(문제아는 오히려 수행, 모범생만 거부).
  CI      = 쿼리 단위 부트스트랩(같은 쿼리의 응답은 한 덩어리로 다시 뽑음). 주장에는 McNemar 보다 이 CI 를 쓴다.
사용법: python3 -I scripts/05_analyze.py data/judged.jsonl [--safe-complete-as-refusal] [--split test]
       [--persona-split test]   (held-out persona template family)
"""
import json
import random
import sys
from collections import defaultdict
from math import comb


def refused(decision, sc_refusal):
    """거부로 칠지 판단. sc_refusal=True 면 safe_complete(부분 수행 + 우회)도 거부로 센다."""
    return decision == "decline" or (sc_refusal and decision == "safe_complete")


def mcnemar_exact(b, c):
    """McNemar 정확검정(양측). b, c = 불일치 쌍의 두 방향 건수. 이항분포 p=0.5 로 계산."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rate(rows, valence, sc):
    """해당 valence 의 거부율."""
    xs = [refused(r["decision"], sc) for r in rows if r["valence"] == valence]
    return sum(xs) / len(xs) if xs else float("nan")


def psg(rows, sc):
    return rate(rows, "problem", sc) - rate(rows, "model", sc)


def bootstrap_ci(rows, sc, n=1000, seed=0):
    """쿼리 단위 부트스트랩으로 PSG 의 95% 신뢰구간을 구한다."""
    rng = random.Random(seed)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    qids = list(by_q)
    stats = []
    for _ in range(n):
        sample = [r for q in rng.choices(qids, k=len(qids)) for r in by_q[q]]  # 쿼리를 복원추출
        d = psg(sample, sc)
        if d == d:  # NaN 제외
            stats.append(d)
    stats.sort()
    return (stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]) if stats else (float("nan"),) * 2


def main(path, sc, split, p_split):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if split:  # 쿼리 분할 필터(보통 test)
        rows = [r for r in rows if r["q_split"] == split]
    if p_split:  # 'none' persona rows have p_split 'all' and stay in every view
        rows = [r for r in rows if r.get("p_split", "all") in ("all", p_split)]
    print(f"{len(rows)} judged generations; safe_complete counted as refusal: {sc}")
    print(f"{'stratum':8s} {'none':>6s} {'neutral':>8s} {'model':>6s} {'problem':>8s} {'PSG':>7s} {'95% CI':>17s} "
          f"{'flip':>6s} {'strict':>7s} {'lenient':>8s} {'McNemar p':>10s}")
    for stratum in sorted({r["stratum"] for r in rows}):
        rs = [r for r in rows if r["stratum"] == stratum]
        # 문제/모범 응답을 짝지을 키: 쿼리, 모델, 표현군, 강도, 채널, 샘플 번호가 모두 같아야 같은 조건의 쌍
        pair_key = lambda r: (r["qid"], r.get("model", ""), r["family"], r["intensity"], r["channel"], r["sample_idx"])
        cells = defaultdict(dict)
        for r in rs:
            if r["valence"] in ("problem", "model"):
                cells[pair_key(r)][r["valence"]] = refused(r["decision"], sc)
        both = [c for c in cells.values() if len(c) == 2]  # 문제/모범이 모두 있는 쌍만
        strict = sum(1 for c in both if c["problem"] and not c["model"])   # problem refused, model complied
        lenient = sum(1 for c in both if c["model"] and not c["problem"])  # model refused, problem complied
        flip = (strict + lenient) / len(both) if both else float("nan")
        lo, hi = bootstrap_ci(rs, sc)
        print(f"{stratum:8s} {rate(rs, 'none', sc):6.3f} {rate(rs, 'neutral', sc):8.3f} {rate(rs, 'model', sc):6.3f} "
              f"{rate(rs, 'problem', sc):8.3f} {psg(rs, sc):+7.3f} [{lo:+6.3f},{hi:+6.3f}] {flip:6.3f} "
              f"{strict:7d} {lenient:8d} {mcnemar_exact(strict, lenient):10.4f}")
    print("PSG > 0: problem-history students are refused more. 'strict' = only the problem persona was refused; "
          "'lenient' = only the model persona was refused.")
    print("Caveat: McNemar treats samples as independent; prefer the query-level bootstrap CI for claims.")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k: a[a.index(k) + 1] if k in a else None   # '--옵션 값' 형태 인자를 꺼내는 간단한 헬퍼
    main(a[0], "--safe-complete-as-refusal" in a, opt("--split"), opt("--persona-split"))
