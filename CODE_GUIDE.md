# 코드 가이드: 구조, 역할, 실행 순서

연구 설계는 [RESEARCH_PLAN.md](RESEARCH_PLAN.md), 이 문서는 **코드가 어떻게 생겼고 어떤 순서로 도는지**만 다룬다.
모든 스크립트 상단 docstring과 본문 주석에 같은 내용이 한국어로 달려 있다.

---

## 1. 한눈에 보는 파이프라인

```
 [원본 데이터]                                   [페르소나]
 EduGuardBench / MinorBench / CASTLE             코드 안의 문장 은행
        │                                              │
        ▼ 00_extract_pools.py                          ▼ 01_build_persona_bank.py
 data/pool_*.jsonl (후보 요청 풀)                 data/personas.jsonl (73개)
        │                                              │
        ▼ 10_label_queries.py  (GPU, LLM 3종 투표)      │
 data/queries.jsonl (정답 행동 + 층 + 분할)             │
        └───────────────────┬──────────────────────────┘
                            ▼ 02_build_probe_grid.py
                 data/grid_<tag>.jsonl (쿼리 x 페르소나 프롬프트)
                            ▼ 11_generate.py   (GPU, 튜터 모델 응답 생성)
                 data/gens_<tag>.jsonl
                            ▼ 12_judge.py      (GPU, 블라인드 판정)
                 data/judged_<tag>.jsonl
                  ┌─────────┴──────────┐
                  ▼                    ▼
        05_analyze.py          03_mine_dpo_pairs.py
        (편향 측정표)           data/dpo_pairs.jsonl (선호쌍)
                                       ▼ 04_train_dpo.py (GPU, LoRA DPO)
                                runs/dpo1 (어댑터)
                                       ▼ 11_generate.py --lora → 12 → 05
                                학습 전후 비교 (편향이 줄었는가)
```

두 가지 흐름이 있다.
- **측정(RQ1·RQ2)**: 10 → 02 → 11 → 12 → 05. "페르소나가 결정을 바꾸는가"
- **교정(RQ3)**: 같은 judged 파일 → 03 → 04 → (다시 11 → 12 → 05). "DPO로 줄어드는가"

---

## 2. 폴더 구조

```
dm-persona/
├── RESEARCH_PLAN.md      연구 설계
├── CODE_GUIDE.md         이 문서
├── data_raw/             내려받은 원본 (건드리지 않음, 신뢰하지 않는 외부 데이터)
├── data/                 모든 중간·최종 산출물 (jsonl)
├── scripts/              파이썬 스크립트 (번호 = 실행 순서)
├── slurm/                슬럼 제출 스크립트 (.sbatch)
├── logs/                 슬럼 job 로그, 분석표
├── .venv-gen/            vLLM 환경 (생성·라벨링·판정)
└── .venv-train/          TRL/PEFT 환경 (DPO 학습)
```

번호 규칙: `00~03`과 `05`는 CPU에서 도는 표준 라이브러리 코드, `04`는 GPU가 필요한 DPO 학습, `10~12`는 vLLM(GPU)이 필요한 코드다.
(번호가 실행 순서와 정확히 일치하지는 않는다. 아래 3장 순서를 따른다.)

---

## 3. 실행 순서

| 순서 | 무엇을 | 명령 | 어디서 | 상태 |
|---|---|---|---|---|
| 0 | 환경 구축 | `sbatch slurm/setup_env.sbatch` | 슬럼(CPU) | 완료 |
| 1 | 후보 풀 추출 | `python3 -I scripts/00_extract_pools.py data_raw data` | 로그인 노드 | 완료 |
| 2 | 페르소나 은행 | `python3 -I scripts/01_build_persona_bank.py data/personas.jsonl` | 로그인 노드 | 완료 |
| 3 | 쿼리 라벨링 | `sbatch slurm/run_label.sbatch` | 슬럼(GPU) | 완료 (736개) |
| 4 | 프로빙 (격자→생성→판정→분석) | `sbatch slurm/run_probe.sbatch <모델> <태그> dev,test 3` | 슬럼(GPU) | **아직 안 돌림** |
| 5 | 선호쌍 채굴 | `python3 -I scripts/03_mine_dpo_pairs.py data/judged_<tag>.jsonl data/dpo_pairs` | 로그인 노드 | 시뮬레이션으로만 검증 |
| 6 | DPO 학습 | `python scripts/04_train_dpo.py --model <id> --pairs data/dpo_pairs.jsonl --out runs/dpo1` | 슬럼(GPU) | 미실행 |
| 7 | 학습 후 재평가 | `sbatch slurm/run_probe.sbatch <모델> <태그>_dpo dev,test 3 runs/dpo1` | 슬럼(GPU) | 미실행 |

`run_probe.sbatch` 한 번이 02 → 11 → 12 → 05를 이어서 실행한다. 모델 하나당 job 하나다.

---

## 4. 파일별 설명

### `scripts/00_extract_pools.py` — 후보 요청 풀 추출
- **역할**: 원본 3종에서 "학생이 튜터에게 하는 요청"만 뽑아 중복을 없앤다.
- **입력/출력**: `data_raw/…` → `data/pool_eduguard.jsonl`(177), `pool_minorbench.jsonl`(299), `pool_castle_misc.jsonl`(5,078)
- **핵심 함수**: `read_xlsx`(pandas 없이 xlsx를 zip+XML로 파싱), `norm`(중복 판정용 정규화)
- **알아둘 점**: EduGuard의 801행은 "교사 프롬프트 × 학생 요청" 조합이라 고유 학생 요청은 177개뿐이다.

### `scripts/01_build_persona_bank.py` — 페르소나 은행
- **역할**: 학생 이력 텍스트를 체계적으로 만든다. 연구의 독립변수.
- **구조**: `INCIDENTS`(도메인×valence별 행동 기록) + `SLOTS`(성적·교사 메모·튜터 이력) + `FAMILIES`(문장 틀 A/B)를 `render()`가 조합한다.
- **조합 수**: 표현군 2 × 도메인 3 × valence 3(모범/문제/무관) × 강도 2(mild 1건, severe 3건) × 학년 2 = 72개, 거기에 이력 없음(`none`) 1개.
- **알아둘 점**: valence끼리 길이 차이가 15% 이하가 되도록 맞췄고(실제 9.4%), 실행하면 그 수치를 출력한다. 표현군 A=train, B=test로 분리해 템플릿 암기를 걸러낸다.

### `scripts/10_label_queries.py` — 정답 행동 라벨링
- **역할**: 각 요청에 대해 "튜터가 해야 할 행동(comply/decline)"을 LLM 3종이 투표해 정한다. 페르소나는 전혀 보여주지 않는다.
- **흐름**: `load_pools` → 모델마다 `run`(라벨링, 실패분은 `RETRY_SYSTEM`으로 재시도, 결과는 `votes_*.json` 캐시) → 투표 집계 → `stratum()`으로 층 결정 → 층당 150개로 자르고 60/15/25로 분할.
- **층**: H(유해), AM-H(부정행위 대행), BL(유해해 보이지만 정상), AM-BL(부정행위처럼 보이지만 정상), C(평범), AMB(모호, 정답 없음).
- **신뢰도**: `unanimous3`(3표 일치) / `unanimous2`(유효 2표 일치) / `split`(불일치). 정확도 주장에는 `unanimous3`만 쓴다.
- **알아둘 점**: 라벨러가 나중에 프로빙 대상 모델이기도 하다. 사람 검수 없이 정확도를 주장하면 안 된다.

### `scripts/02_build_probe_grid.py` — 프롬프트 격자
- **역할**: 쿼리 × 페르소나 조건을 곱해 모델에 넣을 채팅 프롬프트를 만든다.
- **쿼리 하나당 조건**: 이력 없음 1개 + (표현군 2 × valence 3 × 강도 2 × 관련성 2 × 채널 2) = 약 49개.
- **관련성**: 페르소나 도메인과 쿼리 도메인이 같으면 `relevant`(예: 부정행위 이력 × 시험 요청), 다르면 `irrelevant`.
- **채널**: `system_card`(시스템 프롬프트에 기록 카드) / `memory_turns`(튜터가 기록을 불러온다는 가짜 이전 대화).
- **알아둘 점**: 시드 고정이라 모든 모델이 똑같은 격자를 받는다. 튜터 시스템 프롬프트 `TUTOR_SYSTEM`은 모든 조건에서 동일하다.

### `scripts/vllm_util.py` — vLLM 공통 헬퍼
- **역할**: 10·11·12가 공유하는 생성 함수. `chat_generate()`는 대화를 토큰 ID로 직접 렌더링해 vLLM에 넘긴다.
- **왜 필요한가**: vLLM 0.8.5의 `chat()`은 `chat_template_kwargs`가 없어 Qwen3의 thinking을 끌 수 없다. `llm_kwargs()`는 Mistral의 `head_dim=null` 문제를 우회한다.

### `scripts/11_generate.py` — 응답 생성
- **역할**: 격자의 각 프롬프트에 튜터 모델이 답하게 한다. 프롬프트당 `-n`개 샘플.
- **옵션**: `--lora`로 DPO 어댑터를 얹어 학습 후 평가를 한다. `--limit`은 스모크 테스트용.
- **출력**: 격자 행의 메타데이터 + `sample_idx`, `response`, `finish`.

### `scripts/12_judge.py` — 블라인드 판정
- **역할**: 응답을 `comply / safe_complete / decline`으로 분류하고 품질, 훈계 정도, 이력 언급 여부를 채점한다.
- **블라인드**: 판정자는 요청 원문과 응답만 본다. 페르소나를 보면 판정이 이력에 영향받기 때문이다.
- **알아둘 점**: `persona_ref`는 판정자 플래그와 정규식(`HISTORY_RE`)의 OR이다. "warnings" 같은 일반 단어에도 걸릴 수 있어 보수적으로(많이) 잡힌다.

### `scripts/05_analyze.py` — 편향 측정표
- **역할**: 층별로 페르소나가 거부율을 얼마나 바꾸는지 계산한다. 표준 라이브러리만 쓴다.
- **지표**: 이력별 거부율, **PSG**(문제 − 모범 거부율), 쿼리 단위 부트스트랩 95% CI, flip 비율, strict/lenient 건수, McNemar p.
- **알아둘 점**: McNemar는 샘플을 독립으로 보기 때문에 주장에는 부트스트랩 CI를 쓴다. `--split test --persona-split test`로 시험 쿼리와 시험 표현군만 볼 수 있다.

### `scripts/03_mine_dpo_pairs.py` — 선호쌍 채굴
- **역할**: judged 결과에서 DPO용 `(prompt, chosen, rejected)`를 만든다.
- **R1 공유 chosen**: 쿼리마다 정답 행동을 한 최고 품질 응답 y\*를 고르고, 정답을 어긴 프롬프트마다 `(chosen=y*, rejected=실제 위반 응답)` 쌍을 만든다. 페르소나가 달라도 chosen이 같다 → "이력과 무관하게 같은 행동".
- **R2 교차 이식**: 문제/모범 페르소나의 결정이 갈린 곳에서 맞은 쪽 응답을 틀린 쪽 프롬프트의 chosen으로 옮긴다. 편향이 실제로 나타난 어려운 사례만 쓴다.
- **R3 균형**: decline-정답 쌍과 comply-정답 쌍을 1:1로 맞춘다.
- **누수 방지**: train 쿼리, train 표현군(또는 이력 없음)만 쓴다. 이력을 언급한 응답은 chosen 후보에서 뺀다.

### `scripts/04_train_dpo.py` — DPO 학습
- **역할**: TRL `DPOTrainer` + LoRA로 선호쌍을 학습한다. 참조 모델은 어댑터를 끈 기본 모델이다.
- **상태**: 문법만 확인했고 한 번도 실행하지 않았다. TRL 버전에 따라 인자 이름이 다를 수 있다. 반사실 로짓 페어링(CLP) 항은 구현하지 않았다.

### `slurm/*.sbatch`
- `setup_env.sbatch`: 가상환경 2개 구축.
- `run_label.sbatch`: 10번 실행. H200 1장, 3시간 제한.
- `run_probe.sbatch`: 모델 하나에 대해 02 → 11 → 12 → 05를 이어서 실행. 인자: 모델, 태그, 쿼리 split, 샘플 수, (선택) LoRA 경로. 태그별로 파일명이 달라 동시에 여러 개 제출해도 안전하다.

---

## 5. 데이터 형식 (한 줄 = 한 레코드, jsonl)

| 파일 | 주요 필드 |
|---|---|
| `pool_*.jsonl` | `src`, `src_id`, `category`, `query` |
| `personas.jsonl` | `persona_id`, `family`, `valence`, `domain`, `intensity`, `grade`, `split`, `text`, `n_words` |
| `queries.jsonl` | `qid`, `query`, `stratum`, `gold`(decline/comply/none), `domain`, `framing`, `votes`, `confidence`, `split` |
| `grid_<tag>.jsonl` | `qid`, `stratum`, `gold`, `q_split`, `persona_id`, `valence`, `relevance`, `intensity`, `family`, `p_split`, `channel`, `messages` |
| `gens_<tag>.jsonl` | grid 필드 + `model`, `sample_idx`, `response`, `finish` |
| `judged_<tag>.jsonl` | gens 필드 + `decision`, `quality`, `moralizing`, `persona_ref` |
| `dpo_pairs.jsonl` | `prompt`(messages), `chosen`, `rejected` (TRL 대화형). 옆의 `_meta.jsonl`에 레시피·층·페르소나 |

용어: **valence** = 이력의 성격(model 모범 / problem 문제 / neutral 무관 / none 없음), **stratum** = 쿼리 층, **gold** = 튜터가 해야 할 정답 행동, **split** = train/dev/test 분할(쿼리 단위 `q_split`, 페르소나 표현군 단위 `p_split`).

---

## 6. 현재 상태와 검증 수준

- 라벨링 결과(`data/queries.jsonl`, 736개): H 108, AM-H 146, BL 76, AM-BL 150, C 106, AMB 150. 계획 목표(H·BL 각 150)에는 못 미친다. Llama-3가 유해 요청 분류를 거부하는 문제는 재시도로 일부 해결했고 184건은 여전히 투표가 비어 있다.
- 02·03·05는 시뮬레이션 데이터로 검증했다(주입한 편향을 분석기가 되찾는지). **실제 모델 응답으로 돌린 적은 없다.**
- 11은 Qwen3로 3,000건 생성 테스트를 통과했다(약 8,800 tok/s). 12는 아직 한 번도 실행하지 않았다.
- 04는 실행 전이다.

## 작업 위치 (resilience 이전)
- 데이터·venv·HF 모델 캐시·생성물은 모두 `/resilience/ljk98/dm-persona`(HF는 `/resilience/ljk98/hf_cache`)에 둔다. 이 경로는 **계산 노드에서만** 보인다.
- 코드는 로그인 노드의 `/home/ljk98/dm-persona/{scripts,slurm}` 에서 편집한다. 모든 sbatch 는 시작할 때 이 코드를 resilience 로 rsync 한 뒤 거기서 실행한다.
- GPU 작업은 항상 `sbatch slurm/<파일>` 로 제출한다. 분석표(`logs/analysis_*.txt`)는 job 끝에 /home 쪽 logs 로도 복사된다.
- `slurm/migrate.sbatch`: /home 의 기존 data·logs·HF 모델 3종을 resilience 로 복사하는 1회성 job.

## 판정자 검증 (단계 9)
- `scripts/07_make_validation_set.py` + `slurm/run_validation_set.sbatch`: 판정자 decision(comply 60 / safe_complete 80 / decline 60) x 층 x 출처 모델로 층화한 200개를 만든다. 사람에게는 요청+응답만 보이고 페르소나·출처·판정자 결과는 숨긴다.
- `data/judge_validation_200.csv`(라벨링용), `data/judge_validation_GUIDE.md`(기준), `data/judge_validation_key.jsonl`(정답지, 라벨링 전에는 열지 않는다).
- `scripts/08_score_validation.py`: 채운 CSV 와 key 를 비교해 일치율·kappa·혼동행렬·층별 일치율을 낸다(합성 라벨로 동작만 확인).

## 후속 단계 (DPO v2 · 교차검증 · 독립 안전성)
- `03_mine_dpo_pairs.py --hedge-reject`: comply 정답 쿼리의 safe_complete 도 rejected 로 사용(v2). `slurm/run_mine.sbatch` 가 추가 인자를 03 에 넘긴다.
- `slurm/run_judge_xcheck.sbatch`: 검증용 200개를 Llama-3·Mistral 판정자로 다시 판정해 일치율을 본다(결과 `logs/xcheck_judges.txt`).
- `scripts/13_harmbench_check.py` + `slurm/run_harmbench.sbatch`: HarmBench 분류기로 H·AM-H 응답의 실제 유해 수행 비율을 센다(판정자와 독립).
- `scripts/06_compare.py` + `slurm/run_compare.sbatch`: 두 실행의 층·이력별 comply/safe_complete/decline, 품질, 훈계, 이력 언급 비교.

## 감사 이후 보완 (단계 12–15: 32B 판정 · v3/v4 쌍 · 2라운드)
감사(AUDIT_review.md)에서 드러난 판정·쌍 결함을 고친 파이프라인. 결과는 RESULTS_dpo3.md.

| 단계 | 스크립트 / sbatch | 역할 |
|---|---|---|
| 12 재판정 | `12_judge.py --rubric v2 --labels data/labels_v2.json` / `run_rejudge.sbatch "<태그들>"` | Qwen3-32B v2.1 기준으로 다시 판정(출력 `judged_<tag>_q32.jsonl`). labels 가 층·gold 를 덮어쓴다(H→H/HS, AM-H→AM-H/AM-N, AMB 는 gold 없음) |
| 13 gold 재정의 | `17_relabel_gold.py`(32B 초안) → `18_finalize_labels.py`(규칙·출처 기반 확정) | `data/labels_v2.json` 생성 |
| 6 v3 채굴 | `03b_mine_dpo_pairs_v3.py` / `run_mine.sbatch`(MINER 환경변수) | clean() 필터(이력 누출·환각·맞장구·길이 상한 2200, H 는 별도 기준), 중앙 길이 chosen, 쿼리당 8쌍 상한, HS·AM-N 은 짧은 단순 거절만 위반 |
| 6-2 H 합성 위반 | `21_gen_h_violations.py` + `run_hviol.sbatch` | H train 프롬프트에 "경고 없이 답하라"를 덧붙여 base 로 위반 응답 생성(저장 시 원 프롬프트로 복원) → 32B 판정 |
| 6 2라운드 채굴 | `run_mine_r2.sbatch "<judged 태그들>" <접두사>` | base + 이전 DPO(on-policy) train 판정 결과를 합쳐 03b 로 채굴(hviol 은 H 행만) |
| 14 비교 | `run_compare32.sbatch <base> "<태그들>"` (05 PSG 표 + 06 base 대비) | 06 은 HS·AM-N 도 표시 |
| 15 검토 | `19_review_dpo3.py` / `run_review_dpo3.sbatch "<태그들>" <이름>`, `20_h_comply_by_query.py` / `run_hcomply.sbatch`, `run_peek_pairs.sbatch` | 길이·거절문구·이력누출 통계, 층별 샘플 덤프(`logs/review/`), H comply 의 쿼리별 분포, 채굴된 쌍 샘플 |

### 거절 앵커 보강 (단계 6-3~6-5, dpo5)
H(와 AM-H) 거절 쌍이 얇아서 창작·연구 틀 우회에 제동이 약했던 문제를 쿼리 증강으로 보완한다. 결과는 RESULTS_dpo3.md 3라운드.

| 스크립트 / sbatch | 역할 |
|---|---|
| `22_augment_refusal_queries.py` | train 의 H(틀 10가지)·AM-H(앞 3가지) 쿼리를 Qwen3-32B 로 바꿔 써서 `queries_aug.jsonl`(qid `aug-<원본>-<k>`, split=train, gold=decline)과 `labels_aug.json` 생성. dev/test 와 그 변형은 쓰지 않는다 |
| `23_gen_anchor_responses.py` | 쿼리당 격자 프롬프트 `--per-query`(기본 6)개로 위반 응답(`--only viol`, 경고 없이 답하라는 push system 덧붙임, n=2)과 좋은 거절(`--only ref`, 안전 대안이 있는 100~200 단어 거절 지시, n=3)을 생성. 저장하는 messages 는 지시 없는 원 프롬프트. Llama-3 은 거절 지시를 무시해서 ref 는 `--model Qwen/Qwen3-32B` 로 쓴다(off-policy chosen) |
| `24_filter_diag.py` | 03b `clean()` 의 어느 조건에서 응답이 탈락하는지 센다(`python3 -I scripts/24_filter_diag.py data/judged_x.jsonl [층]`) |
| `run_anchor.sbatch` | 위 전체(쿼리 증강 → 격자 → 응답 생성 → 32B 판정 → 앵커 채굴 → 기존 쌍과 병합)를 한 번에 돌린다 |

`run_anchor.sbatch` 사용법: `[옵션 환경변수] sbatch --export=ALL slurm/run_anchor.sbatch <HF base 모델 id> <기존 쌍 접두사> <출력 접두사>`

| 환경변수 | 기본 | 뜻 |
|---|---|---|
| `REUSE=1` | 꺼짐 | 증강 쿼리·격자·위반 응답은 재사용하고 좋은 거절 응답만 다시 만든다 |
| `REF_MODEL` | 첫 인자(base 모델) | 좋은 거절을 쓸 모델(예: `Qwen/Qwen3-32B`) |
| `MINE_ONLY=1` | 꺼짐 | 생성·판정을 건너뛰고 채굴·병합만 다시 한다(GPU 불필요: `-p all --gres=gpu:0` 으로 제출 가능) |
| `H_CAP` | 250 | 병합할 H 앵커 쌍 상한 |
| `AMH_CAP` | 150 | 병합할 AM-H 앵커 쌍 상한. dpo5 는 0 으로 제외 |

앵커는 `03b --no-balance --max-pairs-per-query 3` 로 `data/dpo_aug_anchor` 에 채굴되고, 병합된 쌍에는 recipe=`ANCHOR` 가 붙는다. dpo5 실행: `REUSE=1 REF_MODEL=Qwen/Qwen3-32B` 로 앵커를 만든 뒤 `MINE_ONLY=1 H_CAP=200 AMH_CAP=0` 로 `dpo_llama3_v4` + 앵커 → `dpo_llama3_v5`(782쌍).

### PSG 진단 (단계 16, 결과 RESULTS_psgdiag.md)
`25_psg_diagnostics.py`(numpy 필요, `.venv-gen`) / `run_psgdiag.sbatch "<태그들>" [출력이름]` (CPU, 약 20초): A1 중립 대비 분해, A2 분산 분해(잡음 보정), A3 키워드 분류기, A4 조건별 PSG, A5 BH 보정, A6 쿼리별 PSG 를 `logs/psgdiag.txt` 에 낸다.

실행 예(한 라운드): `run_dpo.sbatch → run_probe.sbatch(LoRA, dev,test) → run_rejudge.sbatch → run_compare32.sbatch / run_hcomply.sbatch / run_harmbench.sbatch / run_review_dpo3.sbatch` 를 `--dependency=afterok` 로 연결.

## 대조 DPO와 쌍대 검정 (단계 17~18, 결과 RESULTS_ctrl5.md)
- `scripts/26_make_control_pairs.py` + `slurm/run_ctrlpairs.sbatch`: 원본 DPO 쌍(예: `dpo_llama3_v5`)에서 마지막 user 턴은 같게 두고 프롬프트를 `valence=none` 프롬프트로 바꿔 `<출력>.jsonl`(+`_meta`)을 만든다. 사용법 `sbatch slurm/run_ctrlpairs.sbatch dpo_llama3_v5 dpo_llama3_v5ctrl`.
- `scripts/27_paired_contrast.py` + `slurm/run_paired.sbatch`: 두 모델의 PSG(문제−모범), 문제−중립, 모범−중립 차이를 같은 쿼리 재표본(2000회)으로 검정하고 BH 보정. 사용법 `sbatch slurm/run_paired.sbatch "llama3_ctrl5:llama3_dpo5 llama3:llama3_ctrl5" paired`, 출력 `logs/paired.txt`. 25번 스크립트의 `load/per_query/contrast/bh`를 재사용한다.
- `slurm/run_du.sbatch`: `/resilience/ljk98` 용량 점검(CPU), 출력 `logs/du_resilience.txt`.

## 스트레스 프로빙 (단계 19)
- `scripts/28_build_stress_grid.py` + `slurm/run_stress.sbatch`: 이력 단서를 가장 강하게 만든 최악 조건 격자. 조건 4종(official 공식 기록·contradict 시점 모순·vague 모호·firstperson 1인칭), 채널 3종(system_card·memory_turns·self_turn). 쿼리는 base의 이력없음/중립 거부율이 0.15~0.85인 것만 고른다(problem/model 조건은 선택에 쓰지 않음). 격자는 `data/grid_stress.jsonl` 한 벌을 모든 모델이 공유한다.
- 사용법: `sbatch slurm/run_stress.sbatch <모델> <태그> [LoRA]`, 판정은 32B v2.1, 출력 `data/judged_<태그>_q32.jsonl`. PSG 진단은 `slurm/run_psgdiag.sbatch`에 태그를 주면 된다(조건 이름은 A4의 "강도" 칸에 나온다).
- 논문에서는 대표 조건 결과와 스트레스 결과를 분리해 보고하고, 스트레스를 전형적 효과로 쓰지 않는다.

## 제안 기법 CPC-DPO (단계 21~22)
- `scripts/30_make_twin_pairs.py` + `slurm/run_twin.sbatch`: 각 선호쌍의 페르소나 문단만 반대 valence(model↔problem, neutral/none은 무작위)로 바꾼 반사실 쌍둥이 프롬프트(`twin_prompt`)를 붙인다. 같은 (표현군, 도메인, 강도, 학년)의 페르소나로 교체해 길이·형식은 같다.
- `scripts/31_train_cpcdpo.py` + `slurm/run_cpc.sbatch`: TRL 없이 PyTorch+PEFT로 구현한 학습. 손실 = 두 프롬프트에서의 DPO 평균 + `lam`·(m_a − m_b)², m 은 내재 보상 마진(β·[(로그비 chosen) − (로그비 rejected)]). `--lam 0` 은 쌍둥이 증강만 있는 절제군(`twinaug`). 하이퍼파라미터는 dpo5와 같다(LoRA r=32/α=64, β 0.1, lr 1e-5, 4 에폭, 유효 배치 32).
- 실행: `sbatch slurm/run_cpc.sbatch <모델> dpo_llama3_v5twin cpc1 --lam 0.05 --seed 0`, 평가 체인 ID는 `logs/chain_cpc.txt`.
- 주의: TRL 구현이 아니라서 처음 실행 때 학습 로그(`logs/dmp-cpc-*.out`)에서 손실이 줄고 |m_a−m_b|가 작아지는지 확인해야 한다.

- `31_train_cpcdpo.py` 수정: `--cons {margin,kl}` 과 `--kl_tokens`(기본 64). `kl` 은 같은 응답에서 두 페르소나 프롬프트의 앞 토큰 다음-토큰 분포 대칭 KL(정책 자체 출력, 참조 무관)을 벌한다. 1차 결과(마진 항이 쌍둥이 증강보다 나쁨)는 `RESULTS_cpc.md`. 실행: `sbatch slurm/run_cpc.sbatch <모델> dpo_llama3_v5twin kl1 --cons kl --lam 1 --seed 0`, 체인 ID 는 `logs/chain_kl.txt`.
- `32_build_stress2_grid.py` + `run_stress2.sbatch`: 학습에 쓰지 않는 새 단서(teacher 쪽지, 12줄 log, 이전 세션 대화 session)의 스트레스 격자 2(`data/grid_stress2.jsonl`). 쿼리 선택은 격자 1과 동일. base 잡이 격자를 만들고 나머지는 그 뒤에 실행(`sbatch --dependency`). 태그는 `llama3_<모델>_s2`.
- `33_make_cueaug_pairs.py` + `run_cueaug.sbatch`: 기존 782 쌍둥이 쌍에 official/vague 단서 쌍둥이 782쌍을 더해 `dpo_llama3_v5cue` 생성(1564쌍). 학습: `cueaug`(lam 0), `cueaug_kl1`(KL lam 1). 보류 단서(firstperson, contradict, teacher, log, session)는 학습에 쓰지 않는다. 체인 ID `logs/chain_cue.txt`.
- `25_psg_diagnostics.py` A7 추가: 스트레스 격자에서 단서별 PSG, 평균 |d_q|, 우연 대비 초과분(AMB). 평균 상쇄를 막는 주 지표.
- `34_build_stress3_grid.py` + `run_stress3.sbatch`: 새 보류 단서 counselor/reportcard/parent 격자(`grid_stress3.jsonl`), 태그 접미사 `_st3`.
- `33_make_cueaug_pairs.py` 확장: `[단서 목록] [벌 수]` 인자. cuemix 쌍은 `sbatch slurm/run_cueaug.sbatch dpo_llama3_v5twin dpo_llama3_v5mix official,vague,teacher,log,firstperson 2`. 체인 `logs/chain_mix.txt`.
