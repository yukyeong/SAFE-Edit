#!/usr/bin/env bash
# Shared helpers for SAFE-Edit entrypoint scripts.
# Usage: source "$(dirname "$0")/../_common.sh"   # from scripts/<subdir>/
#    or: source "$(dirname "$0")/_common.sh"      # from scripts/

_safe_edit_find_repo_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  while [ "${dir}" != "/" ]; do
    if [ -f "${dir}/requirements.txt" ] && [ -d "${dir}/scripts" ]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  echo "SAFE-Edit: could not locate repository root from ${BASH_SOURCE[0]}" >&2
  return 1
}

REPO_ROOT="$(_safe_edit_find_repo_root)"
cd "${REPO_ROOT}"

# Single import path for all Python packages under src/
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

safe_edit_ensure_dirs() {
  mkdir -p \
    "${REPO_ROOT}/outputs" \
    "${REPO_ROOT}/logs" \
    "${REPO_ROOT}/configs/heads"
}
