# 모든 sbatch 가 set -euo pipefail 바로 아래에서 source 한다:  source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
# 경로는 환경변수로만 바꾼다(코드 안에 절대경로 없음).
#   DMP_CODE : 저장소 루트(scripts/, slurm/ 이 있는 곳). 기본값 = sbatch 를 실행한 디렉터리 → 항상 저장소 루트에서 sbatch 한다.
#   DMP_WORK : data/, runs/, logs/ 와 가상환경(.venv-*)을 두는 곳. 기본값 = DMP_CODE.
#              로그인 노드에서 보이지 않는 별도 스토리지를 쓰는 서버라면 slurm/env.local.sh 에서 지정한다(예: slurm/env.local.example.sh).
#   HF_HOME, UV_CACHE_DIR : 모델/패키지 캐시. 기본값 = $DMP_WORK/hf_cache, $DMP_WORK/uv_cache.
DMP_CODE="${DMP_CODE:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$DMP_CODE/slurm/env.local.sh" ] && . "$DMP_CODE/slurm/env.local.sh"
DMP_WORK="${DMP_WORK:-$DMP_CODE}"
export DMP_CODE DMP_WORK
export ROOT="$DMP_WORK"   # 예전 스크립트 호환
export HF_HOME="${HF_HOME:-$DMP_WORK/hf_cache}" UV_CACHE_DIR="${UV_CACHE_DIR:-$DMP_WORK/uv_cache}"
mkdir -p "$DMP_WORK/data" "$DMP_WORK/logs" "$DMP_WORK/runs"
# 코드와 작업 디렉터리가 다르면 job 시작 때 코드를 작업 디렉터리로 동기화한다.
if [ "$DMP_WORK" != "$DMP_CODE" ]; then rsync -a "$DMP_CODE/scripts" "$DMP_CODE/slurm" "$DMP_WORK"/; fi
cd "$DMP_WORK"

# back <작업 디렉터리 기준 상대경로...> : 결과를 저장소(DMP_CODE) 쪽 같은 상대경로로 복사한다. 같은 디렉터리면 아무것도 안 한다.
back() {
  [ "$DMP_WORK" = "$DMP_CODE" ] && return 0
  local f; for f in "$@"; do mkdir -p "$DMP_CODE/$(dirname "$f")"; cp -f "$f" "$DMP_CODE/$f"; done
}
# fetch <상대경로> : 저장소 쪽 파일(예: data/labels_v2.json)을 작업 디렉터리로 복사한다.
fetch() { [ "$DMP_WORK" = "$DMP_CODE" ] || cp -f "$DMP_CODE/$1" "$1"; }
