# CPC-DPO 1차 결과와 설계 수정 (Llama-3-8B-Instruct, 시드 1개)

원자료: logs/psgdiag_cpc.txt, logs/paired_cpc.txt, logs/psgdiag_stress_cpc.txt, logs/harmbench_check.txt, logs/h_comply.txt.
판정 32B v2.1, 거부 = decline + safe_complete, 쿼리 부트스트랩 CI, 쌍대 검정은 BH 보정.

## 1. 문제−모범 PSG (대표 조건, dev+test)

| 층 | base | dpo5 | ctrl5 | cpc1 (마진 항, lam 0.05) | twinaug (lam 0) |
|---|---|---|---|---|---|
| AMB | +0.127 | −0.045 | +0.044 | +0.056 [+0.024,+0.093] | **+0.010 [−0.019,+0.045]** |
| AM-H | +0.122 | −0.001 | +0.021 | +0.013 | +0.006 |
| BL | +0.094 | −0.003 | +0.016 | +0.000 | +0.002 |
| C | +0.079 | −0.001 | +0.024 | +0.011 | +0.004 |
| HS | +0.036 | −0.005 | +0.020 | −0.009 | −0.005 |

- cpc1 − twinaug (AMB) = +0.045 [+0.018,+0.076], q=0.011. **마진 일관성 항은 쌍둥이 증강보다 나쁘다.** 설계가 틀렸다.
- twinaug − dpo5 (AMB) = +0.055 [+0.027,+0.088] 이지만 부호의 의미가 다르다. dpo5 는 −0.045 (역방향 과교정), twinaug 는 0 에 가깝다.
- 스트레스 (AMB, 최악 조건): base +0.235, dpo5 −0.025, ctrl5 +0.120, cpc1 +0.074 [+0.023,+0.132], twinaug +0.023 [−0.012,+0.057]. twinaug 의 쿼리별 sd(d_q)=0.092 는 우연 수준 0.092 와 같다(p=0.49). dpo5 는 0.120 vs 0.086.
- 안전: H comply 0/2058 (cpc1). HarmBench AM-H 위반율 cpc1 0.031, twinaug 0.027 (base 0.204). H: 0.007 / 0.002.
- 대표 조건 AMB 의 쿼리별 이질성은 twinaug 도 남는다(sd 0.140 vs 우연 0.055, d>0 32%, d<0 21%).

## 2. 왜 마진 항이 실패했나 (가설, 확인 전)
1. 마진은 참조 모델 대비 로그비라서 **참조의 페르소나 편향이 그대로 남는다.** (m_a−m_b)² 은 절대 확률의 차이를 줄이지 않는다.
2. 제곱 마진 차이는 큰 쪽 마진을 줄여도 줄어든다. 두 프롬프트 모두 정답 쪽으로 가는 것이 아니라 **마진을 낮추는 방향으로 풀릴 수 있다**(수평화).
3. Garg et al. 의 CLP 는 모델 출력(logit) 자체를 짝지우는 방식이다. 우리는 학습 보상 마진을 짝지웠다.

## 3. 선행 연구를 반영한 수정 (scripts/31 `--cons kl`)
- Butcher(Counterfactual DPO): 반사실 프롬프트를 DPO 쌍으로 쓴다 → twinaug 와 같은 구조. 이 구성만으로 PSG 가 0 근처가 된다는 것이 이번 결과.
- Garg(CLP): 모델 출력을 직접 짝지운다 → 같은 응답을 teacher-forcing 했을 때 두 페르소나 프롬프트의 앞 64개 응답 토큰 다음-토큰 분포의 **대칭 KL** 을 벌한다(참조 모델과 무관). 거절/수락 결정이 앞 토큰에서 갈리므로 앞 토큰에 집중한다.
- Weissburg, Plaza-del-Arco: 역방향 편향과 과교정을 지표로 반드시 같이 본다 → 평균 PSG 대신 PSG, 역방향 쿼리 비율, sd(d_q) 를 같이 보고.
- 제출: kl1 (lam 1), kl5 (lam 5) 시드 0, twinaug 시드 1, 2. 체인 ID 는 logs/chain_kl.txt. lam 은 두 값을 사전 지정해 모두 보고하고 사후 선택하지 않는다.

## 4. 한계
시드 1개(cpc1, twinaug 1차). 같은 782쌍. Llama-3-8B 하나. 판정자 사람 검증 전.

## 5. 스트레스 강화 계획 (2026-10-31 사전 고정)
- 목표: 최악 조건에서 기준선이 흔들리는 환경을 만들고 우리 기법이 버티는지 본다. 단서 문구와 쿼리 선택 기준은 결과를 보기 전에 고정했고, 모든 모델에 같은 격자를 쓴다.
- 학습 단서 vs 보류 단서: 학습에 official, vague 를 넣은 cueaug 를 만들었다(`33`). 평가는 단서별로 분리해 (a) 학습에 쓴 단서 (b) 보류 단서(firstperson, contradict, teacher, log, session)를 따로 보고한다. 보류 단서가 일반화 주장의 근거이고, 학습에 쓴 단서의 개선은 일반화 근거로 쓰지 않는다.
- 규칙(사용자 결정으로 수정, 2026-10-31): 단서 문구·쿼리 선택은 결과를 보고 고쳐도 된다. 데이터셋은 우리가 만드는 것이고 기법은 그에 맞춰 제안한다. 대신 논문에 (1) 격자 설계를 결과를 본 뒤 반복해서 고쳤다는 사실 (2) 학습 단서와 보류 단서의 분리 (3) 고친 이력(버전)을 명시한다. lam 은 사전 지정 값(1, 5)을 모두 보고한다.
- 결과 파일: logs/psgdiag_stress2.txt, logs/psgdiag_stress_cue.txt, logs/psgdiag_cue.txt, logs/paired_cue.txt.

## 6. 데이터 오염 기록 (2026-10-31)
- `llama3_twinaug_s2` 태그가 두 곳에서 쓰였다: (a) twinaug 시드 2 의 대표 격자 프로빙, (b) twinaug 시드 0 의 스트레스 2 평가(`llama3_${t}_s2` 규칙에서 t=twinaug). 둘이 같은 gens/judged 파일을 덮어써서 대표 격자의 시드 2 자료가 소실됐다(n=4032).
- 따라서 psgdiag_kl.txt 의 `llama3_twinaug_s2` 행은 **무효**(HS/AM-H 의 넓은 CI, C 행 없음)이고, psgdiag_stress2.txt 의 `llama3_twinaug_s2` 행은 twinaug **시드 0** 의 스트레스 2 결과다(유효). 시드 2 는 `llama3_twinaug_sd2` 로 다시 돌렸다(logs/chain_sd2.txt).
- 교훈: 스트레스 2 태그 접미사를 `_s2` 에서 `_st2` 로 바꾸는 것이 안전하다. 다음 제출부터 `_st2` 를 쓴다.

## 7. 단서 다양화(cuemix) 계획 (2026-11-01)
- 진단: 쌍둥이 증강은 평균 PSG 를 0 근처로 만들지만 단서별 부호가 갈리고(teacher +, log/session −) 쿼리별 편차가 우연보다 크다. 평균 PSG 는 상쇄로 0 이 되므로 주 지표를 **단서별 PSG 와 평균 |d_q| 의 우연 대비 초과분**(25번 A7)으로 바꾼다.
- 학습 단서(cuemix): official, vague, teacher, log, firstperson(패러프레이즈 3종, 평가 문구와 다름). 원래 782쌍 + 단서 쌍 2벌 = 2346쌍(`dpo_llama3_v5mix`).
- 보류 단서(학습에 쓰지 않음): contradict(격자 1), session(격자 2), counselor, reportcard, parent(격자 3, `34_build_stress3_grid.py`). 일반화 주장은 보류 단서에서만 한다.
- 모델: cuemix(시드 0, 1), cuemix_kl1(KL lam 1). 기준선(dpo5, ctrl5, twinaug, kl1, cueaug)도 격자 3 에 같은 방식으로 돌린다. 체인 ID `logs/chain_mix.txt`. 태그 접미사: 격자 2 는 `_st2`, 격자 3 은 `_st3`(이전의 `_s2` 충돌을 피한다).
- 결과 파일: logs/psgdiag_mix.txt, psgdiag_stress1_mix.txt, psgdiag_stress2_mix.txt, psgdiag_stress3_mix.txt, paired_mix.txt. 단서 요약은 각 파일의 [A7] 표.
- 설계는 결과를 보고 고친다(사용자 결정). 격자 버전: 격자 1 (28) → 격자 2 (32) → 격자 3 (34). 논문에 이 반복을 명시한다.
