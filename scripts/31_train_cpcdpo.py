"""[단계 22] CPC-DPO(Counterfactual Persona-Consistent DPO) — 제안 학습 기법. TRL 없이 PyTorch + PEFT 로 직접 구현한다.

아이디어: 편향 = '같은 응답에 대한 선호 강도(내재 보상 마진)가 페르소나에 따라 달라지는 것'.
  쌍 (x_a, y_c, y_r) 과 그 반사실 쌍둥이 (x_b, y_c, y_r) 에서(x_a, x_b 는 페르소나 문단만 반대 valence)
  내재 보상 마진  m_x = beta * [ (log pi(y_c|x) - log ref(y_c|x)) - (log pi(y_r|x) - log ref(y_r|x)) ]  를 구하고
  loss = -( logsigmoid(m_a) + logsigmoid(m_b) ) / 2  +  lam * (m_a - m_b)^2
  첫 항은 두 페르소나 모두에서 정답 행동을 선호하게 하는 DPO 이고, 둘째 항(반사실 로짓 페어링, CLP 형태)은 마진이 페르소나와 무관하게
  같아지도록 직접 벌한다. lam=0 이면 '쌍둥이 증강만 있는 DPO'(절제 실험)이다.
입력 : --pairs  30_make_twin_pairs.py 의 출력(prompt, chosen, rejected, twin_prompt)
출력 : --out 에 LoRA 어댑터 (11_generate.py --lora 로 그대로 평가)
하이퍼파라미터는 04_train_dpo.py(dpo5 와 같은 설정)와 맞춘다: LoRA r=32/alpha 64, beta 0.1, lr 1e-5, 4 epoch, 유효 배치 32 쌍, 최대 길이 1024.
사용법: python scripts/31_train_cpcdpo.py --model <hf-id> --pairs data/dpo_llama3_v5twin.jsonl --out runs/cpc1 --lam 0.05 --seed 0
"""
import argparse
import json
import math
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

MAX_LEN = 1024


def encode(tok, prompt_msgs, completion):
    p = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=True)
    p = list(p["input_ids"]) if hasattr(p, "keys") else list(p)
    c = tok(completion, add_special_tokens=False).input_ids + [tok.eos_token_id]
    if len(p) >= MAX_LEN - 16:
        return None
    return p, c[: MAX_LEN - len(p)]


def batchify(seqs, pad_id, device):
    """seqs: [(prompt_ids, comp_ids)] -> input_ids, attention_mask, completion mask (오른쪽 패딩)."""
    L = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    cm = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (p, c) in enumerate(seqs):
        n = len(p) + len(c)
        ids[i, :n] = torch.tensor(p + c)
        att[i, :n] = 1
        cm[i, len(p):n] = True
    return ids.to(device), att.to(device), cm.to(device)


def seq_logps(model, ids, att, cm, keep=False):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lsm = F.log_softmax(logits, dim=-1)
    lp = torch.gather(lsm, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    out = (lp * cm[:, 1:]).sum(-1)
    return (out, lsm) if keep else out


def sym_kl(lsm, cm, i, j, k):
    """같은 응답(teacher forcing)에서 시퀀스 i, j 의 앞 k 개 응답 토큰 다음-토큰 분포 사이의 대칭 KL 평균.
    두 시퀀스는 프롬프트의 페르소나 문단만 다르고 응답은 같으므로, 분포 차이는 페르소나 때문이다."""
    pi = cm[i, 1:].nonzero().squeeze(-1)[:k]
    pj = cm[j, 1:].nonzero().squeeze(-1)[:k]
    n = min(len(pi), len(pj))
    la, lb = lsm[i, pi[:n]], lsm[j, pj[:n]]
    return 0.5 * ((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.05, help="마진 일관성 항 가중치. 0 이면 쌍둥이 증강만")
    ap.add_argument("--cons", choices=["margin", "kl"], default="margin",
                    help="일관성 항 종류. margin: (m_a-m_b)^2, kl: 응답 앞 토큰 분포의 대칭 KL(정책 자체 출력끼리, 참조 모델과 무관)")
    ap.add_argument("--kl_tokens", type=int, default=64, help="--cons kl 일 때 비교할 응답 앞 토큰 수")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=32, help="한 옵티마이저 스텝에 쓰는 쌍 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    data = []
    for l in open(a.pairs, encoding="utf-8"):
        r = json.loads(l)
        enc = [encode(tok, r[k], r[v][0]["content"]) for k in ("prompt", "twin_prompt") for v in ("chosen", "rejected")]
        if all(e is not None for e in enc):
            data.append(enc)  # [a+yc, a+yr, b+yc, b+yr]
    print(f"{len(data)} usable pairs", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(data) / a.accum) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, 0, steps)
    model.train()
    step, acc_n, stats = 0, 0, {"dpo": 0.0, "cpc": 0.0, "gap": 0.0, "acc": 0.0, "n": 0}
    for ep in range(a.epochs):
        random.shuffle(data)
        for i, enc in enumerate(data):
            ids, att, cm = batchify(enc, tok.pad_token_id, dev)
            with torch.no_grad(), model.disable_adapter():
                ref = seq_logps(model, ids, att, cm)
            if a.cons == "kl" and a.lam > 0:
                pol, lsm = seq_logps(model, ids, att, cm, keep=True)
            else:
                pol = seq_logps(model, ids, att, cm)
            d = pol - ref  # 내재 보상(로그비) 4개
            m_a = a.beta * (d[0] - d[1])
            m_b = a.beta * (d[2] - d[3])
            l_dpo = -(F.logsigmoid(m_a) + F.logsigmoid(m_b)) / 2
            if a.cons == "kl":
                l_cpc = (sym_kl(lsm, cm, 0, 2, a.kl_tokens) + sym_kl(lsm, cm, 1, 3, a.kl_tokens)) / 2 if a.lam > 0 \
                    else torch.zeros((), device=dev)
            else:
                l_cpc = (m_a - m_b) ** 2
            (( l_dpo + a.lam * l_cpc) / a.accum).backward()
            stats["dpo"] += l_dpo.item(); stats["cpc"] += l_cpc.item(); stats["gap"] += (m_a - m_b).abs().item()
            stats["acc"] += float(m_a > 0) + float(m_b > 0); stats["n"] += 1
            acc_n += 1
            if acc_n == a.accum or i == len(data) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1; acc_n = 0
                if step % 5 == 0 or step == 1:
                    n = stats["n"]
                    print(f"step {step}/{steps} ep {ep} dpo {stats['dpo'] / n:.4f} cpc {stats['cpc'] / n:.4f} "
                          f"|m_a-m_b| {stats['gap'] / n:.3f} acc {stats['acc'] / (2 * n):.3f}", flush=True)
                    stats = {"dpo": 0.0, "cpc": 0.0, "gap": 0.0, "acc": 0.0, "n": 0}
    model.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
