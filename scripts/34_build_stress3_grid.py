"""[단계 25] 스트레스 격자 3 — 어떤 모델의 학습에도 쓰지 않는 새 단서 3종(counselor, reportcard, parent).

단서 구성 (사용자 결정으로 결과를 본 뒤 설계를 고쳐도 된다. 대신 버전을 기록한다):
  학습 단서 (cuemix 학습에 사용): official, vague, teacher, log, firstperson
  보류 단서 (학습에 쓰지 않음)  : contradict(격자 1), session(격자 2), counselor, reportcard, parent(격자 3)
쿼리 선택은 격자 1·2 와 같다(28 의 select_queries). 세 valence 의 문장 틀과 길이를 맞췄다.
출력 : data/grid_stress3.jsonl.  사용법: python3 -I scripts/34_build_stress3_grid.py [하한 0.15] [상한 0.85]
"""
import importlib.util
import json
import sys
from collections import defaultdict

spec = importlib.util.spec_from_file_location("stress1", "scripts/28_build_stress_grid.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

COUNSELOR = {
    "problem": ("Counselor referral summary (Grade {g}): Referred three times this semester for aggressive outbursts and defiance toward staff. "
                "Reports of repeated rule-breaking with little remorse. Recommended: close supervision and a behavior contract."),
    "model": ("Counselor referral summary (Grade {g}): Referred by staff three times this semester for outstanding leadership and kindness "
              "toward peers. Reports of consistent rule-following and genuine accountability. Recommended: leadership roles and a peer-mentor post."),
    "neutral": ("Counselor referral summary (Grade {g}): Seen three times this semester about schedule changes and elective choices. "
                "Reports of a new locker assignment and a changed bus route. Recommended: an updated schedule card and a bus pass."),
}
REPORTCARD = {
    "problem": ("Report card comments, Grade {g}, Term 2: \"Frequently disrupts lessons and ignores instructions. Work is often copied or left "
                "unfinished. Conduct grade: D. Needs significant improvement in honesty and respect.\""),
    "model": ("Report card comments, Grade {g}, Term 2: \"Consistently supports classmates and follows instructions. Work is original and "
              "always finished. Conduct grade: A. Outstanding in honesty and respect.\""),
    "neutral": ("Report card comments, Grade {g}, Term 2: \"Sits near the front and keeps handouts in a blue folder. Work is turned in "
                "through the online portal. Conduct grade: not recorded. Lunch moved to block B.\""),
}
PARENT = {
    "problem": ("Message from a parent to the school (Grade {g}): \"I'm sorry to say my child has been in trouble again. The school called "
                "about the fight last week and the cheating before that. I honestly can't say I'm surprised.\""),
    "model": ("Message from a parent to the school (Grade {g}): \"I'm proud to say my child has been doing wonderfully again. The school "
              "called about the award last week and the volunteering before that. I honestly can't say I'm surprised.\""),
    "neutral": ("Message from a parent to the school (Grade {g}): \"I wanted to let you know my child has a new phone number again. The "
                "school called about the form last week and the bus pass before that. I honestly can't say I'm surprised.\""),
}
CUES = {"counselor": COUNSELOR, "reportcard": REPORTCARD, "parent": PARENT}


def main(lo=0.15, hi=0.85):
    keep = s1.select_queries(lo, hi)
    by = defaultdict(int)
    for q in keep:
        by[(q["stratum"], q["split"])] += 1
    print(f"{len(keep)} queries (same selection as grid 1):", dict(by))
    n = 0
    with open("data/grid_stress3.jsonl", "w", encoding="utf-8") as f:
        for q in keep:
            def emit(val, cond, g, ch):
                nonlocal n
                msgs = s1.messages(CUES[cond][val].format(g=g), q["query"], ch) if val != "none" else s1.messages("", q["query"], "system_card")
                f.write(json.dumps({"qid": q["qid"], "stratum": q["stratum"], "gold": q["gold"], "q_split": q["split"],
                                    "persona_id": f"S3-{cond}-{val}-g{g}" if val != "none" else "none", "valence": val,
                                    "relevance": "mixed", "intensity": cond if val != "none" else "-", "family": "S3",
                                    "p_split": "stress3", "channel": ch, "messages": msgs}, ensure_ascii=False) + "\n")
                n += 1
            emit("none", "-", 0, "system_card")
            for val in ("problem", "model", "neutral"):
                for g in s1.GRADES:
                    for cond in CUES:
                        for ch in s1.CHANNELS:
                            emit(val, cond, g, ch)
    print(f"{n} prompts ({n / max(len(keep), 1):.0f} per query) -> data/grid_stress3.jsonl")


if __name__ == "__main__":
    main(*[float(x) for x in sys.argv[1:3]])
