#!/bin/bash
# [단계 6] 판정된 생성 결과에서 DPO 선호쌍을 채굴한다. CPU 만 쓰는 짧은 job.
# 사용법: sbatch slurm/run_mine.sbatch <judged 태그 예: llama3_train> <출력 접두사 예: dpo_llama3> [03 의 추가 인자, 예: --hedge-reject]
#SBATCH -J dmp-mine
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
python3 -I ${MINER:-scripts/03_mine_dpo_pairs.py} data/judged_$1.jsonl data/$2 "${@:3}" | tee logs/mine_$2.txt
back logs/mine_$2.txt
# 로그인 노드에서 훑어볼 수 있게 메타(분포 확인용)만 /home 에 복사
back data/$2_meta.jsonl
