# Clean Slate: Making LLM Tutors' Guardrails Blind to Student Conduct History

학생의 **과거 품행 기록**(모범/문제 이력)이 LLM 튜터의 **현재 요청에 대한 안전 거절**을 바꾸는지 재고(PSG), 이를 줄이는 학습법을 제안·평가하는 프로젝트.
목표 학회: The Web Conference (WWW) 2027 short track (초록 2026-11-09, 본문 2026-11-16 AoE, OpenReview, 4쪽 참고문헌 포함, 이중 맹검). 대안은 WSDM 2027 short(11-17).

## 핵심 정의
- **PSG** = 거절률(문제 이력) − 거절률(모범 이력). 거절 = decline + safe_complete. 쿼리 단위 부트스트랩 CI, BH 보정, 모델 간은 같은 재표본을 쓰는 쌍대 검정.
- 층: H(유해), HS, AM-H, AM-N, BL, AM-BL, C, AMB (재라벨 후, `data/labels_v2.json`). 판정자는 Qwen3-32B(v2.1 루브릭), 사람 검증은 아직 안 됨.
- 기법: **반사실 쌍둥이 DPO** — 각 선호쌍에 페르소나 문단만 반대 valence 로 바꾼 쌍둥이 프롬프트를 붙여 같은 응답으로 함께 학습(`twinaug`). 변형: 단서 다양화(`cuemix`), 응답 앞 토큰 분포 대칭 KL 일관성(`--cons kl`).

## 지금까지의 결과 (Llama-3-8B-Instruct, 상세는 RESULTS_*.md)
- base PSG(AMB) +0.127, Qwen3 +0.073, Mistral +0.063 (모두 유의). 스트레스 조건에서 Llama-3 는 약 2배(+0.235). "스트레스 2배"는 쿼리를 Llama-3 거부율로 골랐기 때문에 Llama-3 한정 결과.
- 대조 DPO(ctrl5, 페르소나 제거)가 PSG 감소의 큰 부분을 설명. 페르소나 쌍(dpo5)은 AMB 에서 역방향 과교정(−0.045).
- 쌍둥이 증강(twinaug)은 AMB PSG ≈ +0.01 로 0 근처(시드 0, 1 재현), 역방향 과교정 없음. 마진 일관성 항(cpc1)은 오히려 나빴음. KL 항은 일부 지표에서 유망하나 시드 반복 중.
- 현재 지표 한계: 평균 PSG 는 0 근처에서 포화, 쌍둥이 계열 간 차이는 현재 격자(AMB 쿼리 26개)에서 잡음에 묻힘 → **평가 민감도 상향(스트레스 쿼리·샘플 수 증가)이 다음 작업**.

## 구조
- `scripts/` 번호 순서가 파이프라인. 설명은 `CODE_GUIDE.md`. 주요 스크립트: 02 격자, 11 생성(vLLM), 12 판정, 25 PSG 진단(A1~A7), 27 쌍대 대비, 28/32/34 스트레스 격자 1/2/3, 30 쌍둥이 쌍, 31 학습(PyTorch+PEFT), 33 단서 증강 쌍.
- `slurm/` 모든 GPU 작업은 sbatch 로 제출(파티션 h200 GPU / all CPU, 서버에 맞게 `-p` 수정). 경로는 전부 상대경로이고 `slurm/env.sh` 가 `DMP_CODE`(저장소 루트), `DMP_WORK`(data/, runs/, logs/, .venv-* 위치)를 정한다. **sbatch 는 저장소 루트에서 제출한다.**
- `RESULTS_*.md` 결과 기록(날짜순 해석), `RESEARCH_PLAN.md` 설계, `LITERATURE_METHODS.md` 선행 연구와 방법, `AUDIT_review.md` 데이터 감사.
- `logs/*.txt` 결과 요약 로그(psgdiag_*, paired_*, chain_*). 실험 체인 ID 는 `logs/chain_*.txt`.

## 데이터셋 (다른 곳에서 이어서 작업)
원자료 3종은 공개 출처에서 받는다(`bash scripts/fetch_data_raw.sh`, 체크섬 `data_raw/SHA256SUMS`):
- EduGuardBench `Dataset/adversarial_prompts.xlsx` — https://github.com/YL1N/EduGuardBench (arXiv 2511.06890)
- MinorBench `test.csv` — https://huggingface.co/datasets/govtech/MinorBench (arXiv 2503.10242, MIT)
- CASTLE `data/CASTLE_en.json` — https://github.com/ECNU-RAIL/CASTLE-EMNLP2026 (arXiv 2602.05633)
그 다음 `python3 -I scripts/00_extract_pools.py data_raw data` 로 풀을 만든다. 쿼리 라벨링(10)·재라벨(17~18)·페르소나(01)는 LLM 판정이 섞여 재실행하면 값이 달라지므로, **같은 결과로 이어가려면 파생 데이터를 그대로 옮긴다.**
서버에서 `sbatch slurm/pack_data.sbatch` 를 한 번 실행하면 `$DMP_WORK/export/` 에 `dmp_essentials.tar.gz`(쿼리·라벨·페르소나·DPO 쌍·격자), `dmp_results.tar.gz`(판정 결과), `dmp_runs.tar`(LoRA 어댑터만, 약 8GB)와 `SHA256SUMS` 가 생긴다. 이 서버에서는 `/resilience/ljk98/dm-persona/export/` (2026-11-02 생성: essentials 12MB, results 381MB, runs 8.2GB; 로그인 노드에서는 안 보이니 계산 노드나 전송 노드에서 rsync/scp 로 가져간다). 새 위치의 저장소 루트에서 `tar xzf dmp_essentials.tar.gz` 로 풀면 같은 상대경로로 복원된다. 이 묶음에는 모델 응답과 이력 문구가 있어 공개 저장소에는 올리지 않는다.

## 올라와 있지 않은 것
`data/`(응답, 판정, 격자, 쌍), `data_raw/`(CASTLE, EduGuard, MinorBench 원자료), `runs/`(LoRA 어댑터), 가상환경, SLURM 출력, 모델 응답이 담긴 검토 파일.
재현하려면 원자료를 받아 `scripts/00_extract_pools.py` 부터 순서대로 실행한다. 사람이 라벨링할 `judge_validation_200.csv` 는 모델 응답이 섞여 있어 올리지 않았다.

## 이어서 작업하려면
1. 새 서버: 저장소를 clone → (필요하면) `cp slurm/env.local.example.sh slurm/env.local.sh` 로 `DMP_WORK` 등 지정 → 위 데이터셋 절대로 데이터 복원 → `sbatch slurm/setup_env.sbatch` 로 `.venv-gen`(vLLM), `.venv-train`(TRL/PEFT) 생성 → 저장소 루트에서 sbatch. 모델은 `slurm/download_model.sbatch`. 쓰는 모델: NousResearch/Meta-Llama-3-8B-Instruct(튜터), Qwen/Qwen3-32B(판정), cais/HarmBench-Llama-2-13b-cls(안전 점검), Qwen3-8B·Mistral(교차 모델).
2. 실행 중이거나 막 끝난 체인: `logs/chain_mix_seeds.txt`(cuemix, cuemix_kl1 시드 반복). 결과 파일은 `logs/psgdiag_*_seeds.txt`, `paired_mix_seeds.txt`.
3. 다음 작업: (a) 스트레스 쿼리 확대와 샘플 수 증가(평가 민감도) (b) 시드 결과로 KL 항·단서 다양화 채택 여부 결정 (c) 4쪽 초안(WWW sigconf 익명) (d) 판정자 사람 검증 `judge_validation_200.csv`(사용자가 라벨링, 11-09 전).
4. 주의: 격자 1→2→3 은 결과를 본 뒤 반복 설계했다(논문에 명시). 학습 단서와 보류 단서를 분리해 일반화는 보류 단서에서만 주장한다. 태그 충돌 기록은 `RESULTS_cpc.md` 6절.

## 익명성
이 저장소 이름과 계정은 저자를 드러낸다. 투고 시에는 익명 저장소(예: anonymous.4open.science)로 따로 제공하고, 커밋 작성자 정보도 비워야 한다.
