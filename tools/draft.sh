#!/usr/bin/env bash
# Build the padded drafter (tools/pad_draft.py) next to the downloaded GGUFs, once. A no-op when SPEC uses no
# drafter or when the padded file's stamp matches its inputs.
set -euo pipefail

dest=/models
quant=${QUANT:-Q6_K}
target=$dest/Nanbeige_Nanbeige4.3-33B-${quant}.gguf
# Keep these file names in step with tools/download.sh and tools/serve.sh.
source=$dest/Nanbeige4.1-3B.Q4_K_M.gguf
out=$dest/Nanbeige4.1-3B-pad.Q4_K_M.gguf
stamp=$dest/.Nanbeige4.1-3B-pad.Q4_K_M.gguf.inputs

case ${SPEC:-draft+ngram} in
  draft | draft+ngram) ;;
  *) echo "SPEC=${SPEC} uses no drafter"; exit 0 ;;
esac
# The tokenizer is the same in every quant of the target, so any of them serves as the source of its metadata.
inputs="$(<"$dest/.$(basename "$source").sha256") gguf==0.19.0 numpy==2.5.4"
if [[ -f $out && -f $stamp && $(<"$stamp") == "$inputs" ]]; then
  echo "present $(basename "$out")"
  exit 0
fi
rm -f "$out" "$stamp"
uv run --no-project --with gguf==0.19.0 --with numpy==2.5.4 python /pad_draft.py "$target" "$source" "$out.part"
mv "$out.part" "$out"
printf '%s\n' "$inputs" >"$stamp"
echo "built $(basename "$out")"
