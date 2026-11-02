#!/bin/bash
#SBATCH -J dmp-diag
#SBATCH -p all
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /home/ljk98/dm-persona/logs/%x-%j.out
set -euo pipefail
ROOT=/resilience/ljk98/dm-persona
rsync -a /home/ljk98/dm-persona/scripts "$ROOT"/
cd "$ROOT"
for s in H AM-H; do echo "== ref $s"; python3 -I scripts/24_filter_diag.py data/judged_aug_ref_q32.jsonl $s; done
