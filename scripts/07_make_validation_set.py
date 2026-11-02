"""[단계 9] 판정자(Qwen3-8B)를 사람 라벨로 검증하기 위한 층화 표본 200개를 만든다. (stdlib 만 사용)

설계
- 표본 출처: dev+test 쿼리의 판정 결과 4종(Llama-3, Qwen3, Mistral, Llama-3+DPO). 응답 스타일이 다양하게 섞이도록 한다.
- 층화 기준은 '판정자의 decision' x '쿼리 층(stratum)' x '출처 모델'. 판정자 오류는 경계(특히 safe_complete)에서 가장 많을 것이므로
  safe_complete 를 80개, comply/decline 을 60개씩 둔다(자연 비율로 뽑으면 safe_complete 가 거의 안 뽑히는 층이 생긴다).
- 사람 라벨러에게는 판정자와 똑같이 '요청 + 응답'만 보여준다. 페르소나, 출처 모델, 판정자 결과, 정답 라벨은 숨긴다(블라인드).
- 같은 쿼리는 최대 2개까지만 뽑아 쿼리 다양성을 확보한다.
출력: data/judge_validation_200.csv       — 라벨링용 (id, query, reply, human_decision 등 빈칸)
      data/judge_validation_key.jsonl     — 정답지(판정자 결과·페르소나·출처). 라벨링이 끝나기 전에는 열어보지 않는다.
사용법: python3 -I scripts/07_make_validation_set.py data 200 --run llama3 --run qwen3 --run mistral --run llama3_dpo1
"""
import argparse
import csv
import json
import random
from collections import defaultdict

QUOTA_RATIO = {"comply": 0.30, "safe_complete": 0.40, "decline": 0.30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("n", type=int)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    qtext = {}
    for l in open(f"{a.data_dir}/queries.jsonl", encoding="utf-8"):
        q = json.loads(l)
        qtext[q["qid"]] = q["query"]

    # 후보: decision -> (stratum, run) 칸 -> 행 목록. 같은 (run, qid, 응답)은 한 번만.
    cells = defaultdict(lambda: defaultdict(list))
    seen = set()
    for run in a.run:
        for l in open(f"{a.data_dir}/judged_{run}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["q_split"] not in ("dev", "test"):
                continue
            k = (run, r["qid"], r["response"])
            if k in seen:
                continue
            seen.add(k)
            r["run"] = run
            cells[r["decision"]][(r["stratum"], run)].append(r)
    for d in cells:
        for c in cells[d].values():
            rng.shuffle(c)

    chosen, per_q = [], defaultdict(int)
    for d, ratio in QUOTA_RATIO.items():
        quota = round(a.n * ratio)
        keys = sorted(cells[d])           # 칸을 하나씩 돌며 뽑는다 → 층·모델이 고르게 섞인다 (빈 칸은 건너뜀)
        rng.shuffle(keys)
        got = 0
        while got < quota and any(cells[d][k] for k in keys):
            for k in keys:
                while cells[d][k] and per_q[cells[d][k][-1]["qid"]] >= 2:
                    cells[d][k].pop()     # 이미 2개 뽑힌 쿼리는 건너뜀
                if cells[d][k] and got < quota:
                    r = cells[d][k].pop()
                    per_q[r["qid"]] += 1
                    chosen.append(r)
                    got += 1
        print(f"{d}: {got}/{quota}")
    rng.shuffle(chosen)

    with open(f"{a.data_dir}/judge_validation_200.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "reply", "human_decision", "human_quality", "human_moralizing", "notes"])
        for i, r in enumerate(chosen, 1):
            w.writerow([i, qtext[r["qid"]], r["response"][:3000], "", "", "", ""])
    with open(f"{a.data_dir}/judge_validation_key.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(chosen, 1):
            f.write(json.dumps({"id": i, "qid": r["qid"], "run": r["run"], "stratum": r["stratum"], "gold": r["gold"],
                                "valence": r["valence"], "persona_id": r["persona_id"], "judge_decision": r["decision"],
                                "judge_quality": r["quality"], "judge_moralizing": r["moralizing"]}, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for r in chosen:
        by[(r["decision"], r["stratum"])] += 1
    for k in sorted(by):
        print(k, by[k])
    print("runs:", {run: sum(1 for r in chosen if r["run"] == run) for run in a.run}, "| distinct queries:", len(per_q))


if __name__ == "__main__":
    main()
