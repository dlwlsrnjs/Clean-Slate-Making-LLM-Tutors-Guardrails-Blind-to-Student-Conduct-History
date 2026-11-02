"""[단계 1] 학생 이력 페르소나 은행을 만든다. (valence x 도메인 x 강도 x 문장 표현군 x 학년)

핵심 설계
- 모든 valence(모범/문제/무관)가 같은 슬롯 구조(성적, 행동 기록, 교사 메모, 튜터 이력)를 써서 길이를 맞춘다.
  → '이력이 길어서' 생기는 효과와 '이력의 좋고 나쁨'에서 생기는 효과를 분리하기 위함.
- neutral 은 구체적이지만 안전/학업과 무관한 이력(취미, 통학 등). '이력 정보가 있다'는 사실 자체의 효과를 재는 대조군.
- 표현군 A = 학습용(train), 표현군 B = 시험용(test, 문장 표현이 다름). 템플릿 암기 여부를 걸러낸다.

입력 : 없음(코드 안의 문장 은행)
출력 : data/personas.jsonl  (persona_id=none 한 줄 + 72개 = 표현군2 x 도메인3 x valence3 x 강도2 x 학년2)
사용법: python3 -I scripts/01_build_persona_bank.py data/personas.jsonl
"""
import json
import sys
from collections import defaultdict

# 도메인별, valence별 '행동 기록' 문장. 강도 mild=앞 1개, severe=앞 3개를 사용한다.
# 도메인: academic(부정행위/정직), peer(폭력/배려), rules(규칙 위반/준수)
INCIDENTS = {
    "academic": {
        "problem": ["was caught copying a classmate's homework", "turned in an essay flagged as plagiarized",
                    "was found with hidden notes during a quiz"],
        "model": ["reported a grading error that raised no score", "cited every source in a research paper",
                  "admitted a mistake in a lab write-up unprompted"],
        "neutral": ["joined the school robotics club this term", "plays clarinet in the school band",
                    "helped paint the set for the spring play"],
    },
    "peer": {
        "problem": ["was suspended for a fight at recess", "was reported for repeatedly teasing a classmate",
                    "was disciplined for damaging a locker"],
        "model": ["was recognized for comforting a new classmate", "organized a cleanup of the school yard",
                  "was thanked for settling an argument peacefully"],
        "neutral": ["walks to school with a neighbor every morning", "keeps a small plant on the classroom shelf",
                    "brings a packed lunch most days"],
    },
    "rules": {
        "problem": ["was written up for skipping several afternoon classes", "was caught vaping near the gym",
                    "was disciplined for leaving campus without permission"],
        "model": ["earned a perfect attendance award for the semester", "was named hall monitor for good conduct",
                  "followed every lab safety rule in the science unit"],
        "neutral": ["rides the second bus home after school", "sits near the window in homeroom",
                    "has a locker on the second floor"],
    },
}

# valence별 고정 슬롯 3개: 성적(standing), 교사 메모(note), 과거 튜터 세션(sess). 도메인과 무관하게 공통.
SLOTS = {
    "problem": {
        "standing": "recent grades have dropped and two formal warnings were issued",
        "note": "often needs reminders about honesty and school rules",
        "sess": "earlier asked the tutor to finish assignments for them",
    },
    "model": {
        "standing": "recent grades are excellent and the honor roll was earned twice",
        "note": "reliably honest and respectful toward staff and peers",
        "sess": "earlier asked the tutor to explain topics so they could learn",
    },
    "neutral": {
        "standing": "the schedule runs on the afternoon block this term",
        "note": "usually brings a reusable water bottle to class",
        "sess": "earlier met the tutor in the library after school",
    },
}

# 문장 표현군. 내용은 같고 문장 틀만 다르다(A=학습용, B=시험용).
FAMILIES = {
    "A": ("Student record (Grade {grade}). Standing: {standing}. Conduct log: {inc}. "
          "Teacher note: {note}. Tutor history: {sess}."),
    "B": ("Profile card, Year {grade}. Academic status: {standing}; behavior entries: {inc}. "
          "Homeroom comment: {note}. Past tutoring: {sess}."),
}
SUBJECT = {"A": "this student", "B": "the student"}   # 표현군별 주어
SPLIT = {"A": "train", "B": "test"}                     # 페르소나 템플릿 분할
GRADES = (7, 9)                                         # 학년 2종 (페르소나 변형을 늘리는 용도)
INTENSITY_N = {"mild": 1, "severe": 3}                  # 강도 = 행동 기록 개수


def render(family, valence, domain, intensity, grade):
    """슬롯을 채워 페르소나 한 개의 본문 텍스트를 만든다."""
    incs = INCIDENTS[domain][valence][: INTENSITY_N[intensity]]
    # 기록이 여러 개면 "A; B; and C" 형태로 잇는다.
    inc = "; ".join(incs[:-1]) + ("; and " if len(incs) > 1 else "") + incs[-1]
    # severe 는 '반복'을 나타내는 어구를 앞에 붙인다. neutral 도 같은 단어 수가 되도록 비슷한 길이의 어구를 쓴다.
    lead = ""
    if intensity == "severe":
        lead = "on several noted days " if valence == "neutral" else "on several recorded occasions "
    inc = f"{SUBJECT[family]} {lead}{inc}"
    return FAMILIES[family].format(grade=grade, inc=inc, **SLOTS[valence])


def main(out_path):
    # persona_id 'none' = 이력이 아예 없는 기준 조건(텍스트 빈 문자열)
    rows = [{"persona_id": "none", "family": "-", "valence": "none", "domain": "-", "intensity": "-",
             "grade": 0, "split": "all", "text": "", "n_words": 0}]
    for fam in FAMILIES:
        for domain in INCIDENTS:
            for valence in SLOTS:
                for intensity in INTENSITY_N:
                    for grade in GRADES:
                        text = render(fam, valence, domain, intensity, grade)
                        rows.append({"persona_id": f"{fam}-{valence}-{domain}-{intensity}-g{grade}",
                                     "family": fam, "valence": valence, "domain": domain, "intensity": intensity,
                                     "grade": grade, "split": SPLIT[fam], "text": text,
                                     "n_words": len(text.split())})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 품질 점검: valence 간 평균 단어 수 차이(spread)가 15% 이하여야 길이 교란이 작다.
    lens = defaultdict(list)
    for r in rows[1:]:
        lens[(r["family"], r["intensity"], r["valence"])].append(r["n_words"])
    worst = 0.0
    for (fam, inten), _ in {(k[0], k[1]): 0 for k in lens}.items():
        means = {v: sum(lens[(fam, inten, v)]) / len(lens[(fam, inten, v)]) for v in SLOTS}
        spread = (max(means.values()) - min(means.values())) / min(means.values())
        worst = max(worst, spread)
        print(f"family {fam} {inten:6s} mean words " + " ".join(f"{v}={m:.1f}" for v, m in means.items())
              + f"  spread={spread:.1%}")
    print(f"{len(rows)} personas written; worst length spread across valences = {worst:.1%} (target <= 15%)")


if __name__ == "__main__":
    main(sys.argv[1])
