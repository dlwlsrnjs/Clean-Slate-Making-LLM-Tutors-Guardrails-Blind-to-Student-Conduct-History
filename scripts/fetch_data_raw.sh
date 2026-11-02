#!/bin/bash
# 원자료 3종을 data_raw/ 로 내려받는다(저장소 루트에서 실행: bash scripts/fetch_data_raw.sh).
# 이어서 python3 -I scripts/00_extract_pools.py data_raw data  로 풀을 만든다. 체크섬은 data_raw/SHA256SUMS.
# 외부 데이터는 신뢰할 수 없는 입력이다: 열어 보기만 하고 실행하지 않는다. 라이선스는 각 저장소/데이터 카드에서 확인한다.
set -euo pipefail
mkdir -p data_raw/eduguard data_raw/minorbench data_raw/castle
curl -fL -o data_raw/eduguard/adversarial_prompts.xlsx \
  https://raw.githubusercontent.com/YL1N/EduGuardBench/master/Dataset/adversarial_prompts.xlsx
curl -fL -o data_raw/minorbench/test.csv \
  https://huggingface.co/datasets/govtech/MinorBench/resolve/main/test.csv
curl -fL -o data_raw/castle/CASTLE_en.json \
  https://raw.githubusercontent.com/ECNU-RAIL/CASTLE-EMNLP2026/main/data/CASTLE_en.json
(cd data_raw && sha256sum -c SHA256SUMS) && echo "체크섬 일치" || echo "체크섬 불일치: 원본이 갱신됐을 수 있다. 풀 개수(RESEARCH_PLAN.md 표)를 확인할 것"
