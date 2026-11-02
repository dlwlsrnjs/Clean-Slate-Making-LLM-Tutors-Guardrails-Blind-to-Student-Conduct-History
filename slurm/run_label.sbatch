#!/bin/bash
# [단계 2] 쿼리 라벨링 job. GPU 1장에서 라벨러 3종(Llama-3, Qwen3, Mistral)을 차례로 올려 data/queries.jsonl 을 만든다.
# 모델 가중치를 공유 디스크에서 처음 읽을 때 모델당 수 분이 걸려서 제한 시간을 넉넉히(3시간) 잡았다.
# 모델별 투표는 data/votes_*.json 에 캐시되므로, 실패해도 재제출하면 끝난 모델은 건너뛴다.
#SBATCH -J dmp-label
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
# data = 풀이 있는 곳이자 결과(queries.jsonl)를 쓰는 곳. CASTLE 에서 900개, 층당 최대 150개.
.venv-gen/bin/python scripts/10_label_queries.py data data --castle-sample 900 --per-stratum 150
wc -l data/queries.jsonl
