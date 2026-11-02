"""[단계 18] 두 모델의 PSG 차이를 같은 쿼리 재표본으로 직접 검정한다 (numpy 필요, GPU 불필요).

입력 : data/judged_<tag>_q32.jsonl. 거부 = decline + safe_complete. dev+test 쿼리 전체.
출력 : 층별로 PSG(문제−모범)를 모델 A, B 에서 각각 구하고 (A−B) 차이의 쿼리 단위 부트스트랩 CI·p.
       같은 부트스트랩 인덱스를 두 모델에 쓰므로 쿼리 난이도가 상쇄된다(쌍대 검정). 층×비교 전체에 BH 보정을 적용한다.
       PSG 외에 문제−중립, 모범−중립 차이도 같이 낸다.
사용법: python scripts/27_paired_contrast.py <태그A>:<태그B> [<태그A>:<태그B> ...]   예: llama3_ctrl5:llama3_dpo5
"""
import importlib.util
import sys

import numpy as np

# 25 번 스크립트의 로더·집계·BH 를 재사용한다 (파일명이 숫자로 시작해 일반 import 불가).
spec = importlib.util.spec_from_file_location("psgdiag", "scripts/25_psg_diagnostics.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)

B = pd.B
CONTRASTS = {"문제−모범": (3, 2), "문제−중립": (3, 1), "모범−중립": (2, 1)}


def per_stratum(tag):
    rows = pd.load(tag)
    out = {}
    for st in pd.STRATA:
        rs = [r for r in rows if r["stratum"] == st]
        if rs:
            qids, S, N = pd.per_query(rs)
            out[st] = (qids, S, N)
    return out


def main():
    rng = np.random.default_rng(0)
    cache = {}
    for pair in sys.argv[1:]:
        ta, tb = pair.split(":")
        for t in (ta, tb):
            if t not in cache:
                cache[t] = per_stratum(t)
        rows_out, ps = [], []
        for st in pd.STRATA:
            if st not in cache[ta] or st not in cache[tb]:
                continue
            qa, Sa, Na = cache[ta][st]
            qb, Sb, Nb = cache[tb][st]
            if qa != qb:  # 두 모델이 같은 쿼리 집합이어야 쌍대 재표본이 성립
                keep = sorted(set(qa) & set(qb))
                ia = [qa.index(q) for q in keep]
                ib = [qb.index(q) for q in keep]
                Sa, Na, Sb, Nb = Sa[ia], Na[ia], Sb[ib], Nb[ib]
            n = Sa.shape[0]
            for name, (a, b) in CONTRASTS.items():
                est = pd.contrast(Sa, Na, a, b) - pd.contrast(Sb, Nb, a, b)
                stats = []
                for _ in range(B):
                    idx = rng.integers(0, n, n)
                    stats.append(pd.contrast(Sa, Na, a, b, idx) - pd.contrast(Sb, Nb, a, b, idx))
                stats = np.array(stats)
                stats = stats[~np.isnan(stats)]
                if len(stats) == 0 or est != est:
                    continue
                if np.ptp(stats) == 0:   # 분산 0(예: H 층) 은 검정 불가 → BH 에서 제외
                    p = float("nan")
                else:
                    p = min(max(2 * min((stats <= 0).mean(), (stats >= 0).mean()), 1.0 / (B + 1)), 1.0)
                rows_out.append((st, name, pd.contrast(Sa, Na, a, b), pd.contrast(Sb, Nb, a, b), est,
                                 np.quantile(stats, 0.025), np.quantile(stats, 0.975), p))
                ps.append(p)
        qs = pd.bh(ps)
        print(f"\n[쌍대 대비] {ta} − {tb}  (쿼리 부트스트랩 {B}회, q: BH)")
        print(f"{'층':6s} {'대비':8s} {ta:>14s} {tb:>14s} | {'차이':>8s} {'95% CI':>20s} {'p':>7s} {'q':>7s}")
        for (st, name, ea, eb, d, lo, hi, p), q in zip(rows_out, qs):
            print(f"{st:6s} {name:8s} {ea:+14.3f} {eb:+14.3f} | {d:+8.3f} [{lo:+.3f},{hi:+.3f}] {p:7.3f} {q:7.3f}")


if __name__ == "__main__":
    main()
