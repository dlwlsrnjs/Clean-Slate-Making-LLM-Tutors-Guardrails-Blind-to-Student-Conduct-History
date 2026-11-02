#!/bin/bash
# 다른 곳에서 이어서 작업할 수 있게 작업 디렉터리의 파생 데이터를 묶는다(CPU job). 결과는 $DMP_WORK/export/ 에 생긴다.
#   dmp_essentials.tar.gz : 쿼리·라벨·페르소나·DPO 쌍·스트레스 격자(재현과 재학습에 필요한 최소 데이터)
#   dmp_results.tar.gz    : 판정 결과 judged_*_q32.jsonl (PSG 진단/쌍대 검정을 다시 돌릴 수 있다)
#   dmp_runs.tar          : LoRA 어댑터 runs/*(체크포인트 제외, 어댑터 파일만). 약 8GB. 필요한 것만 옮겨도 된다
# 새 위치에서는 저장소 루트에서 `tar xzf dmp_essentials.tar.gz` 처럼 풀면 같은 상대경로로 복원된다.
# 모델 응답과 이력 문구가 담겨 있으므로 공개 저장소에 올리지 않는다. 사용법: sbatch slurm/pack_data.sbatch
#SBATCH -J dmp-pack
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
mkdir -p export
shopt -s nullglob
ess=(data/queries*.jsonl data/labels_v2*.json data/personas.jsonl data/pool_*.jsonl data/dpo_*.jsonl data/grid_stress*.jsonl data/judge_validation_*)
res=(data/judged_*_q32.jsonl)
printf '%s\n' "${ess[@]}" > export/essentials.list
printf '%s\n' "${res[@]}" > export/results.list
tar czf export/dmp_essentials.tar.gz -T export/essentials.list
tar czf export/dmp_results.tar.gz -T export/results.list
rm -f export/dmp_runs.tar
tar cf export/dmp_runs.tar --exclude='checkpoint-*' --exclude='optimizer.pt' --exclude='rng_state*' runs
grep -qx 'data/queries.jsonl' export/essentials.list && echo 'queries.jsonl 포함' || echo '경고: queries.jsonl 없음'
(cd export && sha256sum dmp_*.tar* > SHA256SUMS && ls -la dmp_* && cat SHA256SUMS) | tee logs/pack_data.txt
echo "essentials 파일 수: ${#ess[@]}, results: ${#res[@]}" | tee -a logs/pack_data.txt
back logs/pack_data.txt
