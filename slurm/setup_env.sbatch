#!/bin/bash
# [환경 구축] 계산 노드에서 파이썬 가상환경 2개를 만든다. (로그인 노드에서는 pip 설치가 멈춰서 계산 노드를 쓴다)
#   .venv-gen   : vLLM 0.8.5 — 라벨링/생성/판정 스크립트(10, 11, 12)용
#   .venv-train : TRL + PEFT — DPO 학습 스크립트(04)용
# vLLM 은 이 클러스터에서 이미 동작하는 버전(0.8.5 / torch 2.6)에 맞췄고, 학습 환경은 resolver 에 맡겼다.
# setuptools 는 vLLM 의 triton 이 import 할 때 필요한데 uv venv 에는 기본으로 없어서 명시적으로 설치한다.
#SBATCH -J dmp-env
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
rm -rf .venv-gen .venv-train
uv venv --python 3.11 .venv-gen
uv pip install --python .venv-gen/bin/python "vllm==0.8.5" "transformers<4.58" pandas openpyxl setuptools
uv venv --python 3.11 .venv-train
uv pip install --python .venv-train/bin/python torch "trl==0.19.1" peft datasets accelerate "transformers<4.58" setuptools "pyarrow<21"
# 설치 확인: 버전을 출력한다
.venv-gen/bin/python -c "import vllm, torch; print('gen', vllm.__version__, torch.__version__)"
.venv-train/bin/python -c "import trl, peft, torch; print('train', trl.__version__, peft.__version__, torch.__version__)"
