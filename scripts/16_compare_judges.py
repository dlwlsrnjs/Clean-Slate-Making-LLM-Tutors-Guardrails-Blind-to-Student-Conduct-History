"""[단계 12] 검증용 200개에서 새 판정자(Qwen3-32B, v2 기준)와 기존 판정자(Qwen3-8B, v1 기준)를 비교하고, 서로 다른 항목을 읽기 좋게 덤프한다.
입력 : data/judge_validation_200.csv(요청·응답), data/judge_validation_key.jsonl(8B 판정), data/judged_valset_<tag>.jsonl(새 판정)
출력 : 일치율·혼동행렬을 화면에, 불일치 항목을 <out_md> 에 마크다운으로 쓴다(사람이 어느 쪽이 맞는지 읽고 확인하는 용도).
사용법: python3 -I scripts/16_compare_judges.py data q32 logs/review/judge32_valset.md
"""
import csv
import json
import sys
from collections import Counter

C = ("comply", "safe_complete", "decline")


def main():
    d, tag, out = sys.argv[1:4]
    key = {}
    for l in open(f"{d}/judge_validation_key.jsonl", encoding="utf-8"):
        k = json.loads(l)
        key[k["id"]] = k
    rows = {int(r["id"]): r for r in csv.DictReader(open(f"{d}/judge_validation_200.csv", encoding="utf-8-sig"))}
    new = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/judged_valset_{tag}.jsonl", encoding="utf-8")}
    ids = sorted(new)
    cm = Counter((key[i]["judge_decision"], new[i]["decision"]) for i in ids)
    print(f"n={len(ids)}  3분류 일치 {sum(cm[(c, c)] for c in C) / len(ids):.3f}  "
          f"이분(거부=decline+safe_complete) 일치 {sum((key[i]['judge_decision'] != 'comply') == (new[i]['decision'] != 'comply') for i in ids) / len(ids):.3f}")
    print("행=8B(v1), 열=" + tag + "(v2)")
    for a in C:
        print(f"{a:14s}" + "".join(f"{cm[(a, b)]:8d}" for b in C))
    print("새 판정 분포:", dict(Counter(new[i]["decision"] for i in ids)), "| likely_fabricated:", sum(new[i].get("likely_fabricated", False) for i in ids))
    diff = [i for i in ids if key[i]["judge_decision"] != new[i]["decision"]]
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# 판정자 불일치 {len(diff)}개 / {len(ids)}개 (8B v1 vs {tag} v2)\n")
        for i in diff:
            r, n = rows[i], new[i]
            f.write(f"\n---\n## id {i}  층={key[i]['stratum']}  8B={key[i]['judge_decision']}  {tag}={n['decision']}\n"
                    f"**요청**: {r['query'][:600]}\n\n**응답**: {r['reply'][:1000]}\n\n"
                    f"[{tag}] reason: {n.get('reason', '')} | quality={n['quality']} moralizing={n['moralizing']} fabricated={n.get('likely_fabricated')}\n")


if __name__ == "__main__":
    main()
