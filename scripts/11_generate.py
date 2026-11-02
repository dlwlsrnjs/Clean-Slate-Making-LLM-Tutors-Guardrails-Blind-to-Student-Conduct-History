"""[단계 4] 격자 프롬프트에 대해 튜터 모델의 응답을 vLLM 으로 생성한다.

입력 : data/grid_<tag>.jsonl (02 의 출력, 행마다 messages 포함)
출력 : data/gens_<tag>.jsonl — 프롬프트 1개당 n개 샘플, 한 줄 = 응답 1개 (sample_idx, response, finish 추가)
사용법: python scripts/11_generate.py --model Qwen/Qwen3-8B --grid data/grid.jsonl --out data/gens_qwen3.jsonl -n 5
--lora <디렉터리> 를 주면 DPO 로 학습한 LoRA 어댑터를 --model 위에 얹어서 생성한다(학습 후 평가용).
--limit N 은 앞의 N개 프롬프트만 쓰는 스모크 테스트/처리량 측정용.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=5)                         # 프롬프트당 샘플 수
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke tests)")
    ap.add_argument("--lora")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams        # GPU 노드 전용 의존성이라 함수 안에서 import
    from vllm.lora.request import LoRARequest
    from vllm_util import chat_generate, llm_kwargs

    rows = [json.loads(l) for l in open(a.grid, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    llm = LLM(a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, enable_lora=bool(a.lora), max_lora_rank=32,  # 학습한 LoRA r=32 (vLLM 기본 상한은 16)
              seed=a.seed, **llm_kwargs(a.model))
    sp = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, n=a.n, seed=a.seed)
    # 채팅 템플릿 적용은 vllm_util.chat_generate 가 담당(Qwen3 thinking 끄기, system 역할 미지원 모델 처리)
    outs = chat_generate(llm, [r["messages"] for r in rows], sp, a.model,
                         LoRARequest("dpo", 1, a.lora) if a.lora else None)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            for i, c in enumerate(o.outputs):   # 같은 프롬프트의 n개 샘플을 각각 한 줄로 저장
                d = {k: v for k, v in r.items() if k != "messages"}
                d.update(model=a.model + ("+lora" if a.lora else ""), sample_idx=i, messages=r["messages"],
                         response=c.text.strip(), finish=c.finish_reason)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {sum(len(o.outputs) for o in outs)} generations to {a.out}")


if __name__ == "__main__":
    main()
