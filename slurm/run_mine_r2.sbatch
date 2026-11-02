#!/bin/bash
# [단계 6, v3 2라운드] base 와 이전 DPO(on-policy) 의 train 판정 결과를 합쳐 v3 채굴을 한다. CPU job.
# 사용법: sbatch slurm/run_mine_r2.sbatch "<judged 태그 목록, 예: llama3_train_q32 llama3_dpo3_train_q32>" <출력 접두사>
#SBATCH -J dmp-mine2
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts /home/ljk98/dm-persona/slurm "$ROOT"/
cd "$ROOT"
: > data/judged_$2_merged.jsonl
# *_hviol_* (H 합성 위반 응답)은 재라벨 후에도 H(gold=decline)인 행만 쓴다 — 'warn 하지 마라' 지시로 만든 응답이 HS 등의 chosen 후보가 되면 안 된다
for t in $1; do
  case "$t" in
    *hviol*) python3 -I -c "import json,sys; [sys.stdout.write(l) for l in open(sys.argv[1]) if json.loads(l)['stratum']=='H']" data/judged_$t.jsonl >> data/judged_$2_merged.jsonl ;;
    *) cat data/judged_$t.jsonl >> data/judged_$2_merged.jsonl ;;
  esac
done
python3 -I scripts/03b_mine_dpo_pairs_v3.py data/judged_$2_merged.jsonl data/$2 | tee logs/mine_$2.txt
cp logs/mine_$2.txt /home/ljk98/dm-persona/logs/
cp data/$2_meta.jsonl /home/ljk98/dm-persona/data/
