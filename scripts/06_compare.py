"""[단계 8] DPO 전/후 판정 결과를 같은 쿼리·페르소나 조건에서 나란히 비교한다 (stdlib 만 사용).

입력 : 판정 파일 여러 개 (예: data/judged_llama3.jsonl data/judged_llama3_dpo1.jsonl), 각각 라벨 이름과 함께 --run 으로 준다.
출력 : 층(stratum) x 페르소나 성향(valence)별로 comply / safe_complete / decline 비율, 평균 quality·moralizing, 이력 언급률(persona_ref).
       --q-split / --p-split 으로 시험용 쿼리, 시험용 표현군(B)만 볼 수 있다.
사용법: python3 -I scripts/06_compare.py --run base=data/judged_llama3.jsonl --run dpo1=data/judged_llama3_dpo1.jsonl --q-split test --p-split test
"""
import argparse
import json
from collections import defaultdict

VAL = ("none", "neutral", "model", "problem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path")
    ap.add_argument("--q-split")
    ap.add_argument("--p-split", help="'test' 이면 시험용 표현군 B 와 이력 없음(all) 만")
    a = ap.parse_args()
    for spec in a.run:
        label, path = spec.split("=", 1)
        cell = defaultdict(lambda: defaultdict(list))  # (stratum, valence) -> 지표 -> 값 목록
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if a.q_split and r["q_split"] != a.q_split:
                continue
            if a.p_split and r["p_split"] not in (a.p_split, "all"):
                continue
            c = cell[(r["stratum"], r["valence"])]
            for d in ("comply", "safe_complete", "decline"):
                c[d].append(r["decision"] == d)
            c["quality"].append(r["quality"])
            c["moral"].append(r["moralizing"])
            c["ref"].append(bool(r["persona_ref"]))
        print(f"\n=== {label}  (q_split={a.q_split}, p_split={a.p_split})")
        print("stratum  valence    n     comply safe_c decline | quality moral  ref")
        for st in ("H", "HS", "AM-H", "AM-N", "BL", "AM-BL", "C", "AMB"):
            for v in VAL:
                c = cell.get((st, v))
                if not c:
                    continue
                m = lambda k: sum(c[k]) / len(c[k])
                print(f"{st:7s}  {v:8s} {len(c['comply']):6d}  {m('comply'):.3f}  {m('safe_complete'):.3f}  {m('decline'):.3f}  |"
                      f"  {m('quality'):.2f}   {m('moral'):.2f}  {m('ref'):.3f}")


if __name__ == "__main__":
    main()
