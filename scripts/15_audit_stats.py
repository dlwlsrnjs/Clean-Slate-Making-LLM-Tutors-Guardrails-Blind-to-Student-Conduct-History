"""[단계 11-2] 직접 읽어 본 문제들이 얼마나 퍼져 있는지 수치로 확인한다. (stdlib 만 사용)
  1) DPO 쌍: 층별 개수, chosen/rejected 평균 길이, chosen 이 더 짧은 비율, chosen 의 학년('Grade N'/'Year N') 언급 비율,
     chosen 의 거절 문구 비율.
  2) 생성 결과: 실행(run)별·층별 평균 응답 길이와 거절 문구 비율.
사용법: python3 -I scripts/15_audit_stats.py data dpo_llama3 dpo_llama3_v2 --runs llama3 llama3_dpo1 llama3_dpo2
"""
import json
import re
import sys
from collections import defaultdict

GRADE = re.compile(r"\b(grade|year)\s*\d+\b|\b\d+(st|nd|rd|th)[- ]grader\b", re.I)
REFUSE = re.compile(r"^(i )?(cannot|can't|can not|won't)\b|^i'm (sorry|unable)|\bi (cannot|can't|can not) (provide|assist|help|advise|write|recommend)", re.I)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    d = sys.argv[1]
    pair_sets = [a for a in sys.argv[2:] if not a.startswith("--")]
    runs = sys.argv[sys.argv.index("--runs") + 1:] if "--runs" in sys.argv else []
    pair_sets = [p for p in pair_sets if p not in runs]
    for ps in pair_sets:
        pairs, meta = load(f"{d}/{ps}.jsonl"), load(f"{d}/{ps}_meta.jsonl")
        print(f"\n== {ps}: {len(pairs)} 쌍")
        agg = defaultdict(lambda: defaultdict(list))
        for p, m in zip(pairs, meta):
            c, r = p["chosen"][0]["content"], p["rejected"][0]["content"]
            a = agg[m["stratum"]]
            a["n"].append(1)
            a["clen"].append(len(c)); a["rlen"].append(len(r)); a["short"].append(len(c) < len(r))
            a["grade"].append(bool(GRADE.search(c))); a["crefuse"].append(bool(REFUSE.search(c.strip())))
            a["rrefuse"].append(bool(REFUSE.search(r.strip())))
        print("층      n    chosen길이 rejected길이  chosen이더짧음  chosen학년언급  chosen거절문구  rejected거절문구")
        for st, a in sorted(agg.items()):
            mean = lambda k: sum(a[k]) / len(a[k])
            print(f"{st:6s} {len(a['n']):5d}  {mean('clen'):9.0f}  {mean('rlen'):9.0f}     {mean('short'):.2f}          {mean('grade'):.3f}          {mean('crefuse'):.2f}           {mean('rrefuse'):.2f}")
    if runs:
        print("\n== 생성 결과 (dev+test, 표현군 B+이력없음)  평균 응답 길이(자) / 거절 문구 비율")
        for run in runs:
            agg = defaultdict(lambda: ([], []))
            for r in load(f"{d}/judged_{run}.jsonl"):
                if r["q_split"] in ("dev", "test"):
                    agg[r["stratum"]][0].append(len(r["response"]))
                    agg[r["stratum"]][1].append(bool(REFUSE.search(r["response"].strip())))
            print(f"{run:12s} " + "  ".join(f"{st}:{sum(a)/len(a):.0f}/{sum(b)/len(b):.2f}" for st, (a, b) in sorted(agg.items())))


if __name__ == "__main__":
    main()
