# 선행 연구에서 가져올 편향 유발·측정 기법 (교육 LLM 중심)

검색(2026-10-29)으로 찾은 논문의 방법론 중 우리 파이프라인(요인 프로빙 → 32B 판정 → DPO)에 도입할 만한 것을 정리했다.
**읽은 범위**: 아래 내용은 논문 전문이 아니라 arXiv HTML/초록을 가져와 요약한 것이다. EduFair-Bench 는 초록 수준, "Check My Work?" 는 방법 절,
나머지는 방법 절 요약까지 확인했다. 수치를 논문에 인용하기 전에 원문으로 다시 확인해야 한다. 판정자가 만든 요약이라 틀릴 수 있다.

## 1. 읽은 논문과 우리에게 주는 것

| 논문 | 핵심 방법 | 우리에게 주는 것 |
|---|---|---|
| [ChatGPT Doesn't Trust Chargers Fans: Guardrail Sensitivity in Context](https://arxiv.org/abs/2407.06866) (Li et al.) | 생성된 자기소개(5문장) 페르소나 + 민감 요청. **페르소나 없는 기준선**. 페르소나 집합당 5개 × 다수 요청, ANOVA·2원 ANOVA. 일부 페르소나에서만 거절되는 요청만 남김. GPT-4o 분류기 + 키워드 분류기를 함께 사용 | 가장 가까운 선행. "거절이 페르소나로 달라진다"는 틀 자체가 같다. 분류기 두 종류 병행, 요청×페르소나 상호작용 검정이 도입 대상 |
| [No for Some, Yes for Others: Persona Prompts and Other Sources of False Refusal](https://arxiv.org/abs/2509.08075) (Plaza-del-Arco et al., WiNLP 2025) | 페르소나 15개 × 과제 3개 × 문구 9개, 무작위 짝짓기(Monte Carlo)로 표본 구성. 거절은 문자열 매칭. Wasserstein 민감도 지수와 로지스틱 회귀로 모델·과제·페르소나·문구의 기여를 분해 | 결과: **모델 > 과제 > 페르소나 > 문구** 순으로 영향. 페르소나 효과가 작을 수 있다는 경고이자 분해 분석 틀 |
| [LLMs are Biased Teachers](https://arxiv.org/abs/2410.14012) (Weissburg et al., NAACL Findings 2025) | 교사 역할에 학생 속성을 알려 주고 설명 난이도 선택·생성. 27개 속성, 속성군 안에서 z-정규화, MAB/MDB, 부트스트랩, Friedman. 교사 페르소나 vs 학생 페르소나 대조. temperature 0 | **정방향·역방향 편향이 모두 나옴**(우리 dpo 의 과교정과 같은 현상). 설명된 속성(교사 시점) vs 자기 서술(학생 시점) 대조 설계 |
| [Marked Pedagogies](https://arxiv.org/abs/2603.12471) (Tan et al., LAK26) | 기준 프롬프트에 학생 서술 한 문장만 덧붙임. 비교 프롬프트("The student is White."). Fightin' Words 로그오즈로 표지 단어 추출. 이름만 쓴 조건 | **어떤 표지든 붙이면 반응이 달라진다**(백인 비교 조건도 증가). 이름 단서는 명시적 서술보다 약함. 쓰기 피드백 내용이 학생 라벨에 따라 달라짐 |
| [Sociodemographic Biases in Educational Counselling](https://arxiv.org/abs/2604.25932) (Adamczyk et al.) | 같은 비네트에서 `{student}` 자리만 교체. 중립 통제 + 15개 조건 × **정보 밀도 9단계**(인구통계만 ~ 모순된 지표·서술) × 10개 맥락. t검정 + Benjamini–Hochberg | **정보가 모호할수록 편향이 큼**(평균 절대 편향 0.19 vs 구체적 지표 0.07). 모순 정보 조건과 다중비교 보정 |
| [Check My Work?](https://arxiv.org/abs/2506.10297) | 학생이 정답/오답을 언급하는 조건 5개. 정확도 변화, **뒤집힘(flip) 비율**, 토큰 확률 분석 | 학생의 말에 모델이 끌려가는 정도를 flip 율과 **토큰 확률**로 연속 측정 |
| [EduFair-Bench](https://arxiv.org/abs/2609.12949) | 시뮬레이션 학생 + 튜터 대화, 이름 단서 vs 명시 단서 절제 실험, 튜터·학생 인구통계 충돌 조건, 짝지은 Wilcoxon + 부트스트랩, 판정자를 3인 합의 180턴으로 검증 | 초록 기준: **능력이 높아도 공정성이 높지 않고, 교육용 RL 은 편향을 없애지 않고 옮긴다.** 다중 턴 대화로 확장할 근거 |
| [Bias Runs Deep](https://www.iclr.cc/virtual/2024/poster/17986) (ICLR 2024) | 페르소나를 부여해 문제를 풀게 함 | (초록 수준) 명시적으로는 고정관념을 부정해도 페르소나 부여 시 드러나고, **탈편향 프롬프트는 효과가 거의 없음** → 프롬프트 기반 완화를 DPO 와 비교하는 기준선 |
| [The Illusion of Debiasing](https://arxiv.org/abs/2609.07117) | 페르소나 조향 후 BBQ·BOLD 로 편향 측정 | (검색 요약 수준) 편향이 줄지 않고 **재분배**된다 → 우리 AMB 과교정과 직접 관련 |
| [MinorBench](https://arxiv.org/abs/2503.10242) | 시스템 프롬프트에 "12살" 같은 연령 정보를 넣어 거절 변화 | 연령처럼 **정당한 맥락**은 행동을 바꿔야 하고 과거 이력은 바꾸면 안 된다는 대비가 필요하다 |
| [Quantifying LMs' Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324) (Sclar et al.) | 서식만 바꿔도 성능 격차 최대 76점 → 단일 서식 대신 서식 구간 보고 | PSG 도 서식(템플릿·채널)별 범위로 보고 |

## 2. 도입안 (우리 설계에 매핑)

이미 갖춘 것: 페르소나 없음/중립/모범/문제 4단계, 강도·관련성·채널·템플릿군, 쿼리 수준 부트스트랩, 32B 판정, 판정자 외 독립 검증(HarmBench).

### A. 지금 데이터로 바로 할 수 있는 분석 (GPU 불필요, 추가 생성 없음)
1. **중립 대비 분해.** PSG = 문제 − 모범만 보면 어느 쪽이 움직였는지 모른다. `문제 − 중립`, `모범 − 중립` 을 따로 보고한다(Counselling 의 "집단 − 통제" 방식). dpo5 의 AMB −0.045 가 "모범을 더 거절"인지 "문제를 덜 거절"인지가 여기서 갈린다.
2. **분산 분해.** 쿼리, 템플릿군, 채널, 강도, 페르소나가 거절 변동을 얼마나 설명하는지 로지스틱 회귀/Wasserstein 지수로 쪼갠다(No for Some). 페르소나 효과가 쿼리 효과에 비해 얼마나 작은지를 정직하게 보여 준다.
3. **분류기 이중화.** 32B 판정자 외에 키워드 기반 거절 분류기를 돌려 일치율을 본다(Chargers, No for Some).
4. **서식 구간 보고.** 템플릿군 A/B, 채널별 PSG 범위를 표로 낸다(FormatSpread).
5. **다중비교 보정.** 층×지표 표에 BH 보정을 적용한다.
6. **요청×페르소나 상호작용.** 어떤 쿼리에서 효과가 몰려 있는지(쿼리별 PSG 분포)를 본다. 평균 PSG 가 소수 쿼리에 의존하는지 확인한다.

### B. 프로빙 설계에 추가할 조건 (재생성 필요)
1. **길이·정보량을 맞춘 중립 통제.** Marked Pedagogies 는 어떤 표지든 반응을 바꾼다고 보고했다. 우리 `neutral` 이 문제/모범과 같은 길이·구체성인지 확인하고, 아니면 맞춘 통제("학생은 미술 동아리 소속이다" 등)를 추가한다.
2. **정보 밀도 축.** 모호한 서술("가끔 문제가 있었다")과 구체적·검증 가능한 기록(날짜·건수·처분)을 나눈다. 모호할수록 편향이 큰지 본다.
3. **모순·시점 조건.** "과거에는 문제가 있었으나 최근 1년은 없음", "모범이었으나 최근 징계" 같은 조건. DPO 가 "현재 요청에 집중"하는지를 가장 날카롭게 가르는 조건이다.
4. **자기 서술 채널.** "I got detention last month…"처럼 학생이 직접 말하는 1인칭 단서(Weissburg 의 학생 시점). 현재 system_card·memory_turns 는 제3자 서술이다.
5. **이름·암시 단서 절제.** 명시 서술 vs 암시(예: 교실 상황만 언급)를 비교한다.
6. **정당한 맥락 대조.** 연령·학년(정당한 맥락)과 과거 이력(부당한 맥락)을 같은 쿼리에서 대비한다(MinorBench).

### C. 평가 지표 추가
1. **토큰 확률 기반 연속 지표.** 거절 시작 문구들의 첫 토큰 확률을 teacher-forcing 으로 구해 판정자 없이도 거절 성향을 연속값으로 본다(Check My Work?). 천장·바닥 효과로 PSG 가 0 이 되는 문제(현재 가장 큰 약점)를 완화한다.
2. **뒤집힘(flip) 비율.** 이미 06 에 `flip` 열이 있다. 페어 단위로 모범/문제 간 결정이 바뀌는 비율을 주 지표로 승격한다.
3. **정방향/역방향 구분.** 부호가 있는 PSG 와 절대값(MAB)을 같이 낸다.

### D. 비교 기준선 (DPO 의 가치를 보이려면 필수)
1. **프롬프트 기반 완화**: system 에 "학생의 과거 기록과 무관하게 현재 요청만 보고 판단하라"를 넣은 base (Bias Runs Deep: 효과 없다고 보고됨 → 우리 모델에서도 그런지 확인).
2. **페르소나를 섞지 않은 대조 DPO**: 같은 쌍에서 페르소나 정보를 제거하고 학습. PSG 감소가 페르소나 쌍에서 오는지 가른다.
3. **부작용 감시**: 다른 인구통계/이름 페르소나로 PSG 를 재서 편향이 옮겨 가지 않았는지 확인한다(EduFair: 교육용 RL 은 편향을 이동시킴).

## 3. 권장 순서
1. A(기존 데이터 분석, 반나절) — 가설을 먼저 걸러낸다. 특히 A1·A2·A6.
2. D2 대조 DPO — 핵심 증거.
3. C1 토큰 확률 지표 — 천장·바닥 문제 해결.
4. B3·B4·B2 프로빙 확장 후 base 만 재프로빙해 현상이 어느 조건에서 강한지 확인.
5. D1, D3 기준선·부작용.

## 4. 주의
- 위 요약은 논문 전문이 아닌 가져온 텍스트의 요약이다. EduFair-Bench 와 Illusion of Debiasing 은 초록·검색 요약 수준이며, 2026년 9월자 arXiv ID 는 존재와 내용을 원문으로 재확인해야 한다.
- 선행 연구는 대부분 성별·인종·소득 같은 인구통계 단서를 다뤘다. **"과거 행동 이력"이 안전 거절을 바꾼다**는 설정을 직접 다룬 논문은 이번 검색에서 찾지 못했다(후광/앵커링 연구는 있으나 이력→거절은 아님). 이것이 우리 연구의 새로운 부분일 가능성이 있지만, 검색 범위가 한정적이어서 단정할 수는 없다.
- 선행 연구의 경고: 페르소나 효과는 모델·과제 효과보다 작을 수 있다(No for Some). 우리 효과가 모델·쿼리 효과 대비 어느 정도인지 A2 로 먼저 확인해야 한다.

## 5. WWW short 용 가장 가까운 선행 연구 (2026-10-30 검색, 초록/요약 수준 확인)
1. Li et al., "ChatGPT Doesn't Trust Chargers Fans" (EMNLP 2024), arXiv 2407.06866 — 사용자 자기소개 페르소나가 거절을 바꾼다. 가장 가까운 선행. 차이: 우리는 과거 품행 이력과 교육 맥락, 그리고 완화 학습.
2. Ghandeharioun et al., "Who's asking? User personas and the mechanics of latent misalignment", arXiv 2406.12094 — 인식된 사용자 페르소나가 유해 내용 공개를 좌우, 페르소나 조작이 직접 통제보다 효과적. 차이: 우리는 이력 단서의 valence 와 완화.
3. Zhou et al., "User Persona Subspaces Modulate Refusal Behavior in Language Models" (ICML 2026 MI workshop) — 지속 메모리 챗봇에서 사용자 표상이 안전 행동을 바꾼다. 우리 memory_turns 채널과 직결. 차이: 기계론적 분석 vs 행동 측정·학습.
4. Plaza-del-Arco et al., "No for Some, Yes for Others" (WiNLP 2025), arXiv 2509.08075 — 페르소나 효과는 모델·과제보다 작을 수 있다는 경고.
5. Weissburg et al., "LLMs are Biased Teachers" (NAACL Findings 2025), arXiv 2410.14012 — 교육 개인화 편향 측정·정/역방향 편향.
6. Popovici et al., "The Paternalistic Filter" (2026 preprint, arXiv 2607.11292) — 학생 페르소나별 차등 거절(교육). 정체성 기반이고 이력 기반이 아님. 심사 여부 미확인.
7. 완화 쪽: Butcher, "Aligning LLMs with Counterfactual DPO", arXiv 2401.09566 (반사실 프롬프트를 DPO 에 사용); Causal DPO (EACL 2026 Findings); Garg et al., Counterfactual Logit Pairing (AIES 2019, 기억에 의존한 인용이라 원문 확인 필요).
8. Let Them Down Easy (EMNLP 2025 Findings) — 거절의 응답 방식(부분 이행)이 사용자 경험을 좌우. 우리 safe_complete 판정과 연결.
**이력(과거 행동)이 안전 거절을 바꾸는지를 직접 다룬 논문은 이번 검색에서도 찾지 못했다.** 검색 범위가 얕아 투고 전에 한 번 더 확인해야 한다.
