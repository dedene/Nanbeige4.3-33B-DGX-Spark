#!/usr/bin/env bash
# Fetch the pinned Nanbeige4.3-33B GGUF (QUANT) and the drafter SPEC needs into /models. Resumable, sha256
# checked. A later start is a no-op once the revision stamp matches.
set -euo pipefail

dest=/models
model_id=${MODEL_ID:-bartowski/Nanbeige_Nanbeige4.3-33B-GGUF}
revision=${MODEL_REVISION:-9beb68421fdf7be095093a6888325fa2c83ddd38}
quant=${QUANT:-Q6_K}
spec=${SPEC:-draft+ngram}
file=Nanbeige_Nanbeige4.3-33B-${quant}.gguf
drafter=Nanbeige4.1-3B.Q4_K_M.gguf

# sha256 of each file at the pinned revision (Hugging Face LFS oid).
declare -A SHA=(
  [Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf]=2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37
  [Nanbeige_Nanbeige4.3-33B-Q5_K_M.gguf]=2b10206e3ba0a6a8f88b8c23fdbc49e9efccb3331564e711e4fc2901b18288cc
  [Nanbeige_Nanbeige4.3-33B-Q6_K.gguf]=b11c9f21e69d896a1c20b12f88174f069e0e15623140733a49ff8e47c5541fbc
  [Nanbeige_Nanbeige4.3-33B-Q8_0.gguf]=8523452a975fa67d84a926c9a1bbdbb32b78b2a516dbf14d9360271fc53cd7f1
  [Nanbeige4.1-3B.Q4_K_M.gguf]=0ba28f064c66071646e7959c752851abf02a096f1d37e7d0e443e3502e12ca02
)
# The drafter lives in another repo. Pinned like the weights: repo and revision per file.
declare -A SOURCE=(
  [Nanbeige4.1-3B.Q4_K_M.gguf]="mradermacher/Nanbeige4.1-3B-GGUF c2c9e50a6bffb92c194c286e90cdf794b5657b13"
)

curl_fetch() {
  local url=$1 path=$2
  local -a args=(-fL --retry 8 --retry-delay 2 --retry-all-errors -C - --output "$path")
  [[ -n ${HF_TOKEN:-} ]] && args+=(-H "Authorization: Bearer ${HF_TOKEN}")
  curl "${args[@]}" "$url"
}

fetch() {
  local name=$1 repo=$model_id rev=$revision
  [[ -n ${SOURCE[$name]:-} ]] && read -r repo rev <<<"${SOURCE[$name]}"
  local expect=${SHA[$name]:-}
  if [[ -z $expect ]]; then
    echo "no checksum for ${name}; known files: ${!SHA[*]}" >&2
    exit 1
  fi
  local path=$dest/$name stamp=$dest/.$name.sha256 url="https://huggingface.co/${repo}/resolve/${rev}/${name}"
  if [[ -f $stamp && $(<"$stamp") != "$rev $expect" ]]; then
    echo "revision changed, re-downloading $name"
    rm -f "$path" "$stamp"
  fi
  if [[ -f $path && -f $stamp ]]; then
    echo "present $name"
    return
  fi
  mkdir -p "$dest"
  # A complete file without a stamp (copied in, or an interrupted check) is checked rather than fetched again.
  if [[ ! -f $path ]] || [[ $(sha256sum "$path" | awk 'NR==1 { print $1 }') != "$expect" ]]; then
    echo "downloading $name"
    curl_fetch "$url" "$path"
    if [[ $(sha256sum "$path" | awk 'NR==1 { print $1 }') != "$expect" ]]; then
      echo "checksum mismatch for $name, downloading again" >&2
      rm -f "$path"
      curl_fetch "$url" "$path"
      [[ $(sha256sum "$path" | awk 'NR==1 { print $1 }') == "$expect" ]] || { echo "checksum mismatch for $name" >&2; exit 1; }
    fi
  fi
  printf '%s %s\n' "$rev" "$expect" >"$stamp"
  echo "verified $name"
}

# Keep these file names in step with tools/draft.sh and tools/serve.sh. draft.sh pads the drafter's vocabulary.
fetch "$file"
case $spec in
  draft | draft+ngram) fetch "$drafter" ;;
  ngram | none) ;;
  *) echo "SPEC must be draft, draft+ngram, ngram or none (got $spec)" >&2; exit 1 ;;
esac
exit 0
