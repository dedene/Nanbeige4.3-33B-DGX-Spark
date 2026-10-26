#!/usr/bin/env bash
# llama-server arguments for Nanbeige4.3-33B. Sampling defaults follow the model card's agentic preset
# (temperature 1.0, top_p 0.95); a request can override them. Extra arguments from compose are appended.
set -euo pipefail

quant=${QUANT:-Q6_K}
args=(
  --model "/models/Nanbeige_Nanbeige4.3-33B-${quant}.gguf"
  --alias "${SERVED_NAME:-nanbeige}"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8888}"
  --jinja
  --reasoning-format deepseek
  --n-gpu-layers all
  --flash-attn on
  --ctx-size "${CONTEXT:-262144}"
  --parallel "${PARALLEL:-1}"
  --kv-unified
  --temp "${TEMP:-1.0}"
  --top-p "${TOP_P:-0.95}"
  --top-k "${TOP_K:-0}"
  --min-p "${MIN_P:-0}"
)
[[ -n ${UBATCH:-} ]] && args+=(--ubatch-size "${UBATCH}")
# Speculative decoding. Nanbeige4.1-3B shares the tokenizer and drafts for the 33B; tools/draft.sh pads its
# vocabulary so llama.cpp accepts it. ngram-mod drafts text the reply repeats from the prompt or earlier turns.
draft=(--model-draft /models/Nanbeige4.1-3B-pad.Q4_K_M.gguf --spec-draft-ngl all
  --spec-draft-n-max "${SPEC_DRAFT_MAX:-4}" --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0.6}")
# shellcheck disable=SC2054 # --spec-type takes a comma-separated list
case ${SPEC:-draft+ngram} in
  draft) args+=("${draft[@]}" --spec-type draft-simple) ;;
  draft+ngram) args+=("${draft[@]}" --spec-type draft-simple,ngram-mod) ;;
  ngram) args+=(--spec-type ngram-mod) ;;
  none) ;;
  *) echo "SPEC must be draft, draft+ngram, ngram or none (got ${SPEC})" >&2; exit 1 ;;
esac
[[ -n ${API_KEY:-} ]] && args+=(--api-key "${API_KEY}")

# The image sets LLAMA_ARG_HOST, and we pass --host ourselves.
unset LLAMA_ARG_HOST
exec /app/llama-server "${args[@]}" "$@"
