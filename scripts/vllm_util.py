"""vLLM 0.8.5 용 채팅 생성 헬퍼. 10/11/12 번 스크립트가 공통으로 쓴다.

왜 필요한가: vLLM 0.8.5 의 LLM.chat() 은 chat_template_kwargs 인자가 없어서 Qwen3 의 thinking 을 끌 수 없다.
그래서 프롬프트를 토크나이저로 직접 토큰 ID 로 렌더링해서 LLM.generate 에 넘긴다.
(텍스트로 렌더링한 뒤 generate 에 넘기면 BOS 가 두 번 붙을 수 있어 토큰 ID 로 넘긴다.)
- Qwen3 는 enable_thinking=False 로 렌더링한다.
- system 역할을 거부하는 채팅 템플릿(일부 Mistral 계열)은 system 내용을 첫 user 턴 앞에 합친다.
"""


def _merge_system(messages):
    """system 메시지를 첫 번째 user 메시지 앞에 합쳐서 system 없는 대화로 바꾼다."""
    if messages[0]["role"] != "system":
        return messages
    out = [dict(m) for m in messages[1:]]
    out[0]["content"] = messages[0]["content"] + "\n\n" + out[0]["content"]
    return out


def render_ids(tok, messages, model_name):
    """대화 한 개를 모델의 채팅 템플릿으로 렌더링해 토큰 ID 리스트로 돌려준다."""
    kw = {"enable_thinking": False} if "Qwen3" in model_name else {}
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
    except Exception:
        # 템플릿이 system 역할을 거부하면 합쳐서 다시 시도
        return tok.apply_chat_template(_merge_system(messages), add_generation_prompt=True, tokenize=True, **kw)


def chat_generate(llm, messages_list, sampling_params, model_name, lora_request=None):
    """여러 대화를 한 번에 생성한다. 반환은 vLLM RequestOutput 리스트(입력 순서와 동일)."""
    tok = llm.get_tokenizer()
    prompts = [{"prompt_token_ids": render_ids(tok, m, model_name)} for m in messages_list]
    return llm.generate(prompts, sampling_params, lora_request=lora_request)


def llm_kwargs(model_name):
    """Mistral-7B-Instruct-v0.3 ships head_dim=null, which vLLM 0.8.5 cannot handle with transformers 4.57."""
    return {"hf_overrides": {"head_dim": 128}} if "Mistral-7B" in model_name else {}
