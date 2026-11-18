#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if command -v latexmk >/dev/null 2>&1; then
  for document in main; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error "${document}.tex"
  done
else
  compiler="${TECTONIC_BIN:-tectonic}"
  if ! command -v "$compiler" >/dev/null 2>&1; then
    echo 'Install TeX Live + latexmk, or set TECTONIC_BIN to a Tectonic executable.' >&2
    exit 1
  fi
  flags=(--untrusted --keep-logs --keep-intermediates)
  if [[ "${TECTONIC_OFFLINE:-0}" == 1 ]]; then flags+=(--only-cached); fi
  for document in main; do
    "$compiler" "${flags[@]}" "${document}.tex"
  done
fi
echo 'Built main.pdf: manuscript, references and supplementary material in one document.'
