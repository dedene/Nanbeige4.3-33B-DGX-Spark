# AGENTS.md

Serve Nanbeige4.3-33B (dense Llama architecture, bartowski imatrix GGUF, Q6_K 29 GB) on one DGX Spark with stock
llama.cpp via Docker Compose, with Nanbeige4.1-3B as a speculative drafter.

## Layout
- `compose.yaml`: `download` (curl of the pinned revisions, sha256 checked) -> `draft` (pads the drafter's
  vocabulary, once) -> `nanbeige` (llama-server, healthcheck on `/health`). Knobs are `${VAR:-default}` plus
  `tools/serve.sh`. `.env.example` documents them.
- `tools/download.sh`, `tools/draft.sh` + `tools/pad_draft.py`, `tools/serve.sh`: fetch, pad, launch. File names
  are repeated in all three; keep them in step. `tools/check.py`: functional checks (+ `--needle N`).
- The serving image is the official arm64 CUDA 13 llama.cpp build; `draft` runs in the uv image. No Dockerfile.

## Spark
- Same box as `~/Development/Underdog-Saluki-27B-DGX-Spark` (`ssh zenjoy@10.0.0.158`). Checkout at
  `~/Development/Nanbeige4.3-33B-DGX-Spark`. GGUFs in `~/.cache/nanbeige`.
- GB10, sm_121, aarch64, 128 GB unified (121.6 GiB visible), driver 580, CUDA 13, Compose v5.
- Port 8888 is Saluki's for Hermes. Stop Saluki for benchmarks, start it again afterwards.

## Verify
- `docker compose config --quiet`, `shellcheck tools/*.sh`, `python3 -m py_compile tools/*.py`
- `docker compose up -d --wait`, then `tools/check.py` (all ok).

## Pins
- Image `ghcr.io/ggml-org/llama.cpp:server-cuda13@sha256:2ac42bff86574137dd69b1c8ad85330729d8292e1b4fac0639b67a722c4e9558`
  (b11870, `74bca3950`). uv image `ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:85d4cb1a…`, gguf 0.19.0, numpy 2.5.4.
- Weights `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF@9beb68421fdf7be095093a6888325fa2c83ddd38` (from
  `Nanbeige/Nanbeige4.3-33B@c1093c501cc6`). Drafter `mradermacher/Nanbeige4.1-3B-GGUF@c2c9e50a` Q4_K_M.
  sha256 per file in `tools/download.sh`.

## Model facts
- 33.1B dense, 52 layers, hidden 7168, 56 query / 8 KV heads, head dim 128, rope theta 1.2e8, 512K context.
  f16 KV: 208 KB per token (262K tokens = 54 GB). Text only.
- Chat template: `<think>` opened by the generation prompt; `chat_template_kwargs.enable_thinking: false` closes it.
  `preserve_thinking` defaults to keeping past reasoning (the card recommends that for agents). Tool calls use the
  Qwen3-Coder XML format; llama.cpp's `--jinja` parses them.
- Card sampling: agentic temperature 1.0 / top_p 0.95, chat and reasoning 0.6 / 0.95.

## Findings (2026-10-25, bring-up)
- `tools/check.py` all ok on Q6_K: thinking off/on, tool call, tool round trip, streaming, Dutch. West Flanders'
  capital: Kortrijk (should be Brugge), like Flint.
- Decode without drafting (thinking off, 400 tokens): Q4_K_M 10, Q5_K_M 8, Q6_K 7, Q8_0 6 tok/s. Bandwidth-bound.
- The drafter is refused as is: llama.cpp allows a vocab size difference of 128 (`SPEC_VOCAB_MAX_SIZE_DIFFERENCE`),
  and the GGUFs differ by 768 padding entries (166,912 vs 166,144). Tokens 0..166,143 are identical, so
  `pad_draft.py` copies the target's tokenizer metadata and zero-pads token_embd/output. No translation fallback
  in b11870.
- Probe with the padded drafter, n-max 4 (code / English / Dutch tok/s): Q6_K 16/11/9 with the Q4_K_M drafter,
  14/10/9 with Q8_0. p-min 0.6 vs 0 is the same single-stream and drafts less. ngram-mod alone: no gain on fresh text.
- KLD to BF16 (`llama perplexity`, 30 x 2048 tokens of Dutch/English Wikipedia and Python, mean KLD / same top-1):
  Q8_0 0.006 / 97.6%, Q6_K 0.010 / 96.6%, Q5_K_M 0.033 / 93.5%, Q4_K_M 0.062 / 90.8%. Q6_K is the default.

## Findings (2026-10-25, speed sweep)
- Q6_K, 4 slots x 64K, kit speed suite 2 reps (weighted decode / tool call / prefill 41K / 4-stream aggregate):
  none 7.3/7.8/494/27.9, draft 12.6/24.4/420/23.6, draft+ngram 12.6/26.5/429/25.0, draft+ubatch 2048
  12.6/25.0/471/24.4. Defaults: draft+ngram, p-min 0.6, ubatch 2048.
- The drafter loses throughput with 4 requests in flight (draft-simple per slot batches badly) and slows prefill
  (the drafter prefills too). The kit's 32K-character prompt is 41K tokens with this tokenizer.
- Memory with 4 x 64K slots and the drafter: 100,247 MiB on the GPU, `free` 106 of 121 GiB used. Draft KV follows
  the target's context (64 KB per token).
- `docker compose up` runs `draft` in about 60 s the first time (uv fetches gguf + numpy into `/models/.uv-cache`).
  Padded Q4_K_M drafter sha256 `5c85fab935a72ed1f734fe0c5016dd03fd5c5676eded6e05a7a6ceb0b7284d88`.
- `--kv-unified` is on: slots share one KV pool instead of CONTEXT/PARALLEL each. Without it an AgentDojo sample hit
  the 64K per-slot limit (65,558 tokens; this tokenizer needs ~25% more tokens than Qwen's for the same text).
  Benchmark restarted at 23:34 with it; BFCL single-turn (short prompts) kept from before.
- Benchmark (thinking off): BFCL simple 94.0, multiple 90.0, parallel 91.0, parallel-multiple 77.0, irrelevance
  71.0; AgentDojo benign 79.2, under attack 47.5, attacks succeeded 83/200 (banking 14/23, slack 26/29, travel
  22/35, workspace 21/113). Spot-checked: real transfers to the attacker IBAN US133000000121212121212 and scheduled
  payments redirected to it. Agent loop 5.7 steps/min (wait 2-8.5 s per step up to 30K tokens).
- Benchmark finished 2026-10-26 15:43 (report in the kit: `report-nanbeige/`): IFEval 74.7, Global-MMLU Dutch 70.3,
  GPQA-Diamond 64.6 (10 answers hit the 4,096-token cap, all wrong), BFCL multi-turn 56.5. Agentic score 70.7 vs
  81.2 for Qwen FP4-Local. Quality suites ran with `SPEC=none` (lossless, faster with 4 requests in flight).
- DSpark (DFlash + Markov and confidence heads, `--spec-type draft-dspark`) needs a drafter trained on the target's
  hidden states. Nanbeige only ships one for 4.2-3B; none for 4.3-33B yet. A trained one would be the next speedup.

## Constraints
- Stock llama.cpp. Never run `hf cache prune` on the Spark (it deleted pinned models on 2026-10-16).
- Credits: write our own scripts. Underdog-Saluki-27B-DGX-Spark is the pattern.
