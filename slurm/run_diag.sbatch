#!/bin/bash
#SBATCH -J dmp-diag
#SBATCH -p all
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
for s in H AM-H; do echo "== ref $s"; python3 -I scripts/24_filter_diag.py data/judged_aug_ref_q32.jsonl $s; done
