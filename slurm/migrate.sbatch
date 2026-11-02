#!/bin/bash
# [이전] /home 의 데이터·HF 모델을 /resilience/ljk98 로 복사한다. 로그인 노드에서는 /resilience 가 안 보여서 job 으로 한다.
# 프로빙 job(44892)이 끝난 뒤 --dependency=afterany:44892 로 제출해 완성된 data 를 가져간다.
#SBATCH -J dmp-migrate
#SBATCH -p all
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
SRC=/home/ljk98/dm-persona; DST=/resilience/ljk98/dm-persona
mkdir -p $DST /resilience/ljk98/hf_cache/hub
rsync -a $SRC/scripts $SRC/slurm $SRC/*.md $SRC/data_raw $DST/
rsync -a $SRC/data $SRC/logs $DST/
H=/home/ljk98/.cache/huggingface
for m in models--NousResearch--Meta-Llama-3-8B-Instruct models--Qwen--Qwen3-8B models--mistralai--Mistral-7B-Instruct-v0.3; do
  rsync -a $H/hub/$m /resilience/ljk98/hf_cache/hub/
done
du -sh $DST/* /resilience/ljk98/hf_cache/hub/*
