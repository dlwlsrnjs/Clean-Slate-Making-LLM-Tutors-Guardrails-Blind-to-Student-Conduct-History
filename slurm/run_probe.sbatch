#!/bin/bash
# [단계 3~5] 프로빙 한 번 = 격자 만들기 → 응답 생성 → 블라인드 판정 → 분석표 출력. 모델 하나당 job 하나.
# 사용법: sbatch slurm/run_probe.sbatch <HF 모델 id> <태그> <쿼리 split 목록 예: dev,test> <프롬프트당 샘플 수> [LoRA 디렉터리]
# 예)   sbatch slurm/run_probe.sbatch Qwen/Qwen3-8B qwen3 dev,test 3
#       DPO 후 평가: sbatch slurm/run_probe.sbatch Qwen/Qwen3-8B qwen3_dpo dev,test 3 runs/dpo1
# 태그마다 파일 이름이 달라서 여러 job 을 동시에 제출해도 서로 덮어쓰지 않는다.
#SBATCH -J dmp-probe
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
export HF_HOME=/resilience/ljk98/hf_cache      # 모델 캐시도 resilience 에 둔다
export UV_CACHE_DIR=/resilience/ljk98/uv_cache
mkdir -p "$ROOT/data" "$ROOT/logs"
# 코드는 로그인 노드(/home/ljk98/dm-persona)에서 편집하고, job 시작 때 resilience 로 동기화한다 (로그인 노드에서는 /resilience 가 안 보인다)
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
MODEL=$1; TAG=$2; SPLITS=$3; N=$4; LORA=${5:-}
# (1) 라벨링된 쿼리 중 지정한 split 만 골라 이 태그 전용 파일로 저장 (python3 -I = 격리 모드, 표준 라이브러리만 사용)
python3 -I - "$SPLITS" "$TAG" <<'PY'
import json, sys
keep = set(sys.argv[1].split(","))
with open(f"data/queries_probe_{sys.argv[2]}.jsonl", "w") as f:
    for l in open("data/queries.jsonl"):
        if json.loads(l)["split"] in keep:
            f.write(l)
PY
# (2) 쿼리 x 페르소나 격자 프롬프트 생성
python3 -I scripts/02_build_probe_grid.py data/queries_probe_$TAG.jsonl data/personas.jsonl data/grid_$TAG.jsonl
# (3) 튜터 모델 응답 생성 (LoRA 인자가 있을 때만 --lora 를 붙인다)
.venv-gen/bin/python scripts/11_generate.py --model "$MODEL" --grid data/grid_$TAG.jsonl --out data/gens_$TAG.jsonl -n "$N" ${LORA:+--lora "$LORA"}
# (4) 블라인드 판정 → (5) 분석표를 화면과 logs/analysis_<tag>.txt 에 저장
.venv-gen/bin/python scripts/12_judge.py --gens data/gens_$TAG.jsonl --out data/judged_$TAG.jsonl
python3 -I scripts/05_analyze.py data/judged_$TAG.jsonl --safe-complete-as-refusal | tee logs/analysis_$TAG.txt
# 분석표는 로그인 노드에서 읽을 수 있게 /home 쪽 logs 에도 복사해 둔다 (원본 데이터는 resilience 에만 둔다)
cp logs/analysis_$TAG.txt /home/ljk98/dm-persona/logs/
