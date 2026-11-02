"""[단계 6-3] 거절 앵커용 쿼리 증강. train 의 거절 정답(H, AM-H) 쿼리를 여러 '틀'로 바꿔 쓴 새 쿼리를 만든다. (vLLM, Qwen3-32B)

왜: dpo3/dpo4 는 H train 쿼리가 13개뿐이라(쌍 11개) "소설·연구 목적이면 응한다"는 일반화를 막지 못했다.
    같은 요청을 창작·연구·가설·안전교육 등 다른 틀로 바꿔 쓰면 '틀이 바뀌어도 요청 자체가 같으면 같은 판단'을 가르칠 수 있다.
누수 방지: 원본이 train 인 쿼리만 쓴다(dev/test 쿼리나 그 변형은 쓰지 않는다).
입력 : data/queries.jsonl, data/labels_v2.json
출력 : data/queries_aug.jsonl  — queries.jsonl 과 같은 형식(qid="aug-<원본>-<k>", split="train", gold="decline", stratum=H 또는 AM-H)
       data/labels_aug.json    — labels_v2.json + 새 qid (12_judge.py --labels 에 그대로 쓴다)
사용법: python scripts/22_augment_refusal_queries.py data/queries.jsonl data/labels_v2.json [--model Qwen/Qwen3-32B]
"""
import argparse
import json
import re

FRAMES = {
    "fiction": "a scene in a novel or short story the student is writing",
    "webcomic": "a storyline for a webcomic, video game or school play",
    "research": "a school research project or thesis that needs the details",
    "safety_training": "building a safety-awareness or red-team training module for classmates or staff",
    "hypothetical": "a purely hypothetical 'what if' question the student is curious about",
    "teacher": "something a teacher or parent supposedly asked the student to find out",
    "friend": "helping a friend who supposedly urgently needs it",
    "debate": "preparing for a class debate or presentation",
    "roleplay": "a roleplay in which the tutor plays an expert character",
    "journalism": "an investigative school-newspaper article",
}
H_FRAMES = list(FRAMES)          # H(운영형 유해 요청)는 10가지 틀
AMH_FRAMES = list(FRAMES)[:3]    # AM-H(과제 대필 등)는 3가지 틀

PROMPT = """Rewrite the student message below so that it keeps the SAME underlying request, but presents it as {frame}.
Keep it in first person, natural, 1-3 sentences, like a real student. Do not add disclaimers or safety remarks. Do not make the request milder or more harmful than the original.

Original message:
\"\"\"{query}\"\"\"

Answer with one JSON object and nothing else: {{"query": "<rewritten message>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("labels")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", default="data/queries_aug.jsonl")
    ap.add_argument("--out-labels", default="data/labels_aug.json")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm_util import chat_generate, llm_kwargs

    labels = json.load(open(a.labels, encoding="utf-8"))
    todo = []
    for l in open(a.queries, encoding="utf-8"):
        q = json.loads(l)
        lab = labels.get(q["qid"])
        st = lab["stratum"] if lab else q["stratum"]
        gold = lab["gold"] if lab else q["gold"]
        if q["split"] == "train" and gold == "decline" and st in ("H", "AM-H"):
            for k, f in enumerate(H_FRAMES if st == "H" else AMH_FRAMES):
                todo.append((q, st, k, f))
    llm = LLM(a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, **llm_kwargs(a.model))
    msgs = [[{"role": "user", "content": PROMPT.format(frame=FRAMES[f], query=q["query"][:1500])}] for q, _, _, f in todo]
    outs = chat_generate(llm, msgs, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=250, seed=0), a.model)
    new_labels, bad, n = dict(labels), 0, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for (q, st, k, fr), o in zip(todo, outs):
            m = re.search(r"\{.*\}", o.outputs[0].text, re.S)
            try:
                text = json.loads(m.group(0))["query"].strip()
                assert 20 <= len(text) <= 1200
            except Exception:
                bad += 1
                continue
            qid = f"aug-{q['qid']}-{k}"
            row = dict(q, qid=qid, query=text, src="aug", stratum=st, gold="decline", framing=fr, votes=[], confidence="aug",
                       split="train", parent=q["qid"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            new_labels[qid] = {"stratum": st, "gold": "decline", "kind": "aug", "reason": f"augmented from {q['qid']} ({fr})", "old_stratum": st}
            n += 1
    json.dump(new_labels, open(a.out_labels, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"augmented {n} queries from {len({q['qid'] for q, *_ in todo})} originals; unparsable {bad}")


if __name__ == "__main__":
    main()
