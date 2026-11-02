"""[단계 0] 원본 데이터셋 3종에서 '학생 요청' 후보 풀을 뽑아 중복 제거 후 jsonl로 저장한다.

입력 : data_raw/eduguard/adversarial_prompts.xlsx, data_raw/minorbench/test.csv, data_raw/castle/CASTLE_en.json
출력 : data/pool_eduguard.jsonl, data/pool_minorbench.jsonl, data/pool_castle_misc.jsonl
       각 줄 = {"src", "src_id", "category", "query"}

pandas/openpyxl 없이 표준 라이브러리만 쓴다(xlsx는 zip+XML을 직접 파싱).
원본 파일은 신뢰할 수 없는 외부 데이터이므로 `python3 -I`(격리 모드)로 실행한다.
사용법: python3 -I scripts/00_extract_pools.py data_raw data
"""
import csv
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"  # xlsx XML 네임스페이스


def read_xlsx(path):
    """xlsx 첫 시트를 [{헤더: 값}, ...] 로 읽는다. 문자열 셀은 sharedStrings 테이블의 인덱스로 저장돼 있다."""
    z = zipfile.ZipFile(path)
    shared = [
        "".join(t.text or "" for t in si.iter(M + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(M + "si")
    ]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(M + "row"):
        row = {}
        for c in r.findall(M + "c"):
            v = c.find(M + "v")
            if v is not None:
                # 셀 주소 'B12' 에서 숫자를 떼어 열 문자('B')만 키로 쓴다. t="s" 면 공유 문자열 참조.
                row[re.sub(r"\d", "", c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(row)
    header, body = rows[0], rows[1:]
    return [{header[k]: v for k, v in r.items() if k in header} for r in body]


def norm(s):
    """중복 판정용 정규화: 소문자화 + 연속 공백 하나로."""
    return re.sub(r"\s+", " ", s.strip().lower())


def main(raw, out):
    raw, out = Path(raw), Path(out)
    out.mkdir(exist_ok=True)

    # --- EduGuardBench: 801행은 '교사 프롬프트 x 학생 요청' 조합이라 학생 요청 기준으로 중복 제거해야 한다(177개) ---
    edu = read_xlsx(raw / "eduguard/adversarial_prompts.xlsx")
    seen, pool = set(), []
    for r in edu:
        key = norm(r["Student_Statement_EN"])
        if key in seen:
            continue
        seen.add(key)
        # ID 접두어(AM/AP/HR/TS/PI)를 category 로 쓴다. AM=학업부정, 나머지=일반 유해(구실 프레임 많음)
        pool.append({"src": "eduguard", "src_id": r["ID"], "category": r["ID"].split("-")[0],
                     "query": r["Student_Statement_EN"].strip()})
    print("eduguard rows", len(edu), "unique student statements", len(pool),
          "categories", Counter(p["category"] for p in pool))
    write(out / "pool_eduguard.jsonl", pool)

    # --- MinorBench: 299개, 범주 6개. 정상 질문이 섞여 있어 라벨링이 따로 필요 ---
    with open(raw / "minorbench/test.csv", newline="", encoding="utf-8") as f:
        mb = [{"src": "minorbench", "src_id": r["PromptID"], "category": r["Category"], "query": r["Prompt"].strip()}
              for r in csv.DictReader(f)]
    print("minorbench", len(mb), Counter(p["category"] for p in mb))
    write(out / "pool_minorbench.jsonl", mb)

    # --- CASTLE: 3.9만 행 중 '학업부정/부적절 콘텐츠' 하위유형만 쓰고, 질문 기준 중복 제거 ---
    castle = json.load(open(raw / "castle/CASTLE_en.json", encoding="utf-8"))
    sub = [c for c in castle if c["Scenario_Subtype"] in ("Academic Misconduct", "Inappropriate Content")]
    uq = {}
    for c in sub:
        uq.setdefault(norm(c["User_Query"]), {"src": "castle", "src_id": str(len(uq)),
                                              "category": c["Scenario_Subtype"], "query": c["User_Query"].strip()})
    print("castle rows", len(castle), "misconduct/inappropriate rows", len(sub), "unique queries", len(uq))
    write(out / "pool_castle_misc.jsonl", list(uq.values()))


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
