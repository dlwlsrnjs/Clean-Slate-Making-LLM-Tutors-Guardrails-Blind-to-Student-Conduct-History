"""[단계 7] 채굴한 선호쌍으로 TRL DPOTrainer + LoRA 학습을 한다. TRL 0.19.1 기준.

입력 : --model(HF 모델 id), --pairs(03 이 만든 jsonl)
출력 : --out 디렉터리에 LoRA 어댑터 체크포인트. 이후 11_generate.py --lora 로 vLLM 에 얹어 재평가한다.

주의
- TRL 은 버전마다 인자 이름(rpo_alpha, processing_class 등)이 달라진다. 실행 전 설치된 버전(.venv-train)에서 확인할 것.
- 반사실 로짓 페어링(CLP) 정규화 항(RESEARCH_PLAN.md 5.1 R4)은 구현하지 않았다.
- LoRA(peft_config)를 주면 참조 모델은 '어댑터를 끈 기본 모델'이라 참조 모델을 따로 올리지 않는다(메모리 절약).
사용법: python scripts/04_train_dpo.py --model <hf-id> --pairs data/dpo_pairs.jsonl --out runs/dpo1
"""
import argparse

import json

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)       # DPO 의 참조모델 이탈 페널티 강도
    ap.add_argument("--lr", type=float, default=5e-6)        # LoRA 용 학습률(전체 미세조정이면 훨씬 낮게)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rpo-alpha", type=float, default=0.0, help="weight of the chosen-NLL term; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:          # Llama-3 에는 pad 토큰이 없다 → eos 로 대체
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="bfloat16")
    # load_dataset("json") 은 이 환경의 datasets/fsspec 조합에서 실패할 수 있어 jsonl 을 직접 읽는다
    # 채팅 템플릿 적용과 턴 종료 토큰은 TRL(0.19)이 대화형 prompt/chosen/rejected 를 보고 알아서 처리한다
    ds = Dataset.from_list([json.loads(l) for l in open(a.pairs, encoding="utf-8")])

    # 실효 배치 = 2 x 16 = 32. 최대 길이 1024 토큰.
    cfg = DPOConfig(output_dir=a.out, beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
                    per_device_train_batch_size=2, gradient_accumulation_steps=16, max_length=1024,
                    rpo_alpha=a.rpo_alpha or None, bf16=True, logging_steps=10, save_strategy="epoch", seed=a.seed)
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # With a peft config the reference model is the adapter-disabled base, so no separate ref copy is loaded.
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft)
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
