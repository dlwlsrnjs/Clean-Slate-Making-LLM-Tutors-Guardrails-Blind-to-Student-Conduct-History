#!/bin/bash
# 용량 점검(CPU). /resilience/ljk98 아래 항목별 크기를 하나씩 logs/du_resilience.txt 에 이어 쓴다(중간에 끊겨도 앞부분이 남는다).
#SBATCH -J dmp-du
#SBATCH -p all
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 03:00:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
OUT=/home/ljk98/dm-persona/logs/du_resilience.txt
: > "$OUT"
df -h /resilience | tail -1 >> "$OUT"
cd /resilience/ljk98
for d in dm-persona/data dm-persona/logs dm-persona/models dm-persona/outputs dm-persona/.venv-gen dm-persona/.venv-train hf_cache; do
  [ -e "$d" ] && du -sh "$d" >> "$OUT" 2>&1
done
echo "--- dm-persona 최상위" >> "$OUT"
ls -1 dm-persona >> "$OUT"
echo "--- ljk98 최상위" >> "$OUT"
ls -1 . >> "$OUT"
echo DONE >> "$OUT"
