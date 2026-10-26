# Nanbeige4.3-33B on a single DGX Spark (llama.cpp, Docker Compose)

Serve [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) on one NVIDIA DGX Spark with stock llama.cpp. One command downloads a pinned Q6_K GGUF and a small drafter model, prepares the drafter, and starts an OpenAI-compatible API with reasoning and tool calls.

Nanbeige4.3-33B is a dense 33B model for agentic coding and office work, with a switchable reasoning mode and a 512K context window. Dense matters on the Spark: every token reads all 29 GB of weights through about 273 GB/s of memory bandwidth, so plain decoding runs at about 7 tok/s. Speculative decoding with Nanbeige's own 3B model brings one request to 12.6 tok/s on average and up to 26 tok/s on tool calls, without changing the output.

**Hardware:** DGX Spark (GB10, sm_121, aarch64), driver 580, CUDA 13, Docker with the NVIDIA Container Toolkit and Compose v2. About 35 GB free disk. With the full 262K context the server takes about 100 GB of the Spark's memory.

## Quickstart

```bash
git clone https://github.com/dedene/Nanbeige4.3-33B-DGX-Spark.git
cd Nanbeige4.3-33B-DGX-Spark
docker compose up -d --wait     # download 31.5 GB (first run only), pad the drafter, load, wait until healthy
tools/check.py                  # model id, thinking off/on, tool call and round trip, streaming, Dutch
docker compose down
```

The downloads check their sha256; `up --wait` returns once `/health` answers.

The API is at `http://<spark-ip>:8888/v1`, model `nanbeige`. llama-server answers whatever model id a client sends.

```bash
curl http://<spark-ip>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nanbeige",
  "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Thinking is on by default. The reasoning comes back in `message.reasoning_content`; `"chat_template_kwargs": {"enable_thinking": false}` turns it off for a request. The template keeps earlier reasoning in multi-turn conversations, which the model card recommends for agents and tool use. The server samples with the card's agentic preset (temperature 1.0, top_p 0.95). For chat and reasoning the card suggests temperature 0.6.

## How it is put together

- **`compose.yaml`:** `download` fetches the pinned GGUF and drafter and checks them, `draft` prepares the drafter once, then `nanbeige` serves. A healthcheck hits `/health`.
- **`tools/download.sh`:** curl of the pinned Hugging Face revisions, resumable, sha256 checked.
- **`tools/draft.sh`, `tools/pad_draft.py`:** make Nanbeige4.1-3B usable as a drafter (see below). Runs in the uv image with pinned `gguf` and `numpy`.
- **`tools/serve.sh`:** `llama-server` with all layers on the GPU, flash attention, `--jinja` for the chat template and tool calls, the drafter, and one KV pool shared by all slots (`--kv-unified`), so one long agent session can use more than its share of `CONTEXT`.
- **`tools/check.py`:** functional checks, plus `--needle N` for a long-context recall test.
- The image is the official arm64 CUDA 13 build of llama.cpp (b11870), pinned by digest. There is no Dockerfile.

### The drafter

Nanbeige4.1-3B uses the same tokenizer as the 33B, so it can propose tokens for the 33B to check. llama.cpp refuses it as is: the two GGUFs pad the vocabulary to different sizes (166,912 and 166,144 entries), and llama.cpp allows a difference of 128. The first 166,144 tokens are identical. `pad_draft.py` gives the 3B the 33B's tokenizer metadata and adds zero rows for the padding ids, which never occur in real text. The 33B verifies every drafted token, so answers are the same as without the drafter. Nothing is uploaded: the padded file is built locally from the pinned download.

## Configuration

Copy `.env.example` to `.env` and uncomment what you want to change, then `docker compose up -d --wait` again.

| Variable | Default | |
|---|---|---|
| `QUANT` | `Q6_K` | `Q4_K_M`, `Q5_K_M`, `Q6_K` or `Q8_0`. See the quality table below. |
| `SPEC` | `draft+ngram` | `draft`, `ngram` or `none`. All lossless. |
| `SPEC_DRAFT_MAX`, `SPEC_DRAFT_P_MIN` | 4, 0.6 | Tokens drafted per step, and the drafter confidence below which it stops. |
| `CONTEXT` | 262144 | Context llama-server allocates (native: 524288). Slots share it. |
| `PARALLEL` | 1 | Server slots, for concurrent requests. |
| `UBATCH` | 2048 | Prefill micro-batch. |
| `TEMP`, `TOP_P`, `TOP_K`, `MIN_P` | 1.0, 0.95, 0, 0 | Server defaults; a request can override them. |
| `PORT`, `HOST`, `SERVED_NAME` | 8888, 0.0.0.0, nanbeige | |
| `API_KEY` | empty | Bearer token, when set. |
| `LLAMA_CPP_IMAGE`, `MODEL_REVISION` | pinned | Re-run `tools/check.py` after a bump. |

## Which quant

KL divergence to the BF16 GGUF on about 60,000 tokens of Dutch and English Wikipedia and Python code (`llama perplexity`, 30 chunks of 2,048 tokens). Lower is closer to the original. Speed is one request with the drafter, thinking off.

| Quant | Size | Mean KLD | Same top token as BF16 | Code / English / Dutch, tok/s |
|---|---:|---:|---:|---|
| Q8_0 | 35.2 GB | 0.006 | 97.6% | 15 / 10 / 8 |
| **Q6_K** (default) | 29.0 GB | 0.010 | 96.6% | 16 / 11 / 9 |
| Q5_K_M | 25.1 GB | 0.033 | 93.5% | 18 / 12 / 10 |
| Q4_K_M | 20.8 GB | 0.062 | 90.8% | 19 / 14 / 11 |

Q6_K stays close to the original. One step down, Q5_K_M triples the divergence for about 10% more speed.

## Speed tuning

Q6_K, four slots of 64K, the kit's speed suite with 2 repetitions, thinking off. Decode is weighted over code, prose, a tool call and a long prompt.

| Setting | Decode, tok/s | Tool call, tok/s | Dutch, tok/s | Prefill at 41K tokens, tok/s | 4 requests at once, tok/s |
|---|---:|---:|---:|---:|---:|
| No drafting | 7.3 | 7.8 | 7.5 | **494** | **27.9** |
| 3B drafter | 12.6 | 24.4 | 9.8 | 420 | 23.6 |
| 3B drafter + n-gram | **12.6** | **26.5** | **10.1** | 429 | 25.0 |
| 3B drafter, micro-batch 2048 | 12.6 | 25.0 | 9.8 | 471 | 24.4 |

The drafter makes one request 70% faster, and tool calls three times as fast. With four requests at once it costs about 15% of the throughput, and it slows prefill a little, because the drafter processes the prompt too. For a single agent or chat user, keep the default. For a server with many parallel users, `SPEC=none` is faster overall. Drafter settings that didn't help: Q8_0 instead of Q4_K_M for the 3B (slower), 6 drafted tokens instead of 4 (same).

## Notes

- **Memory.** Q6_K with the full 262K window: 29 GB of weights, 54 GB of KV cache (208 KB per token), 2.4 GB of drafter plus 17 GB for its own cache (64 KB per token). Measured: 100,247 MiB. Every 64K tokens of `CONTEXT` less saves about 18 GB, so `CONTEXT=65536` fits next to another model.
- **Text only.** Images in a request are replaced by a note from the chat template.
- **The API is open on the LAN** unless you set `API_KEY`.

## As an agent: benchmark results

I ran this stack through the same agent benchmark as the other models on this Spark, on 25 and 26 October 2026. Same box, same samples, temperature 0, thinking off, 4,096 output tokens, four requests in flight. The suites are the standard [Inspect](https://inspect.aisi.org.uk/) tasks from inspect_evals 0.24.0. Brackets are the 95% interval. The speed rows and the agent loop ran with the default settings (3B drafter + n-gram). The quality suites ran with `SPEC=none`: drafting doesn't change the answers, and without it four parallel requests finish sooner.

| | Nanbeige4.3-33B (Q6_K, llama.cpp + 3B drafter) | Qwen3.8 Flash Next (NVIDIA FP4-Local, vLLM + MTP) | Qwen3.8 Flash Next (MLX 4-bit, TensorFold + MTP) | Recursive Flint (NVFP4 GGUF, llama.cpp) | Ornith-1.5-122B (NVFP4, vLLM + MTP) | Nemotron 3.5 Super VL (NVFP4, vLLM + LiLiCorr) |
|---|---:|---:|---:|---:|---:|---:|
| BFCL single-turn, n=500 | 84.6 (81–87) | 82.4 | 71.8 | 80.0 | **86.0** | 85.6 |
| BFCL multi-turn, n=200 | **56.5** (50–63) | 54.5 | 43.0 | 32.0 | 51.5 | 49.2 |
| AgentDojo, tasks done without an attack, n=96 | 79.2 (70–86) | **84.4** | 78.1 | 75.0 | 81.2 | 77.1 |
| AgentDojo, attacks resisted, n=200 | 58.5 (52–65) | **99.0** | 97.0 | 94.5 | 87.0 | 81.5 |
| AgentDojo, tasks done under attack, n=200 | 47.5 (41–54) | **82.0** | 70.5 | 70.5 | 71.5 | 64.0 |
| IFEval, prompt strict, n=541 | 74.7 (71–78) | 85.8 | 85.0 | **94.6** | 80.2 | 85.6 |
| Global-MMLU Dutch, n=300 | 70.3 (65–75) | 87.0 | **87.7** | 79.3 | 85.7 | 86.7 |
| GPQA-Diamond, n=198 | 64.6 (58–71) | 64.6 | 59.1 | 68.7 | **72.7** | 56.1 |
| Agent loop, steps per minute (8 tool steps) | 5.7 | 13.8 | 21.0 | **24.8** | 8.4 | 19.2 |
| Decode, tok/s weighted over code, prose and tool calls | 12.3 | 43.6 | 67.2 | **92.7** | 37.0 | 34.6 |
| Decode, Dutch prose, tok/s | 9.5 | 31.7 | 50.2 | **96.1** | 33.1 | 23.6 |
| Prefill at ~32k tokens, tok/s | 469 | 2,274 | 2,185 | 1,835 | 2,082 | **2,373** |
| 4 requests at once, tok/s | 23.6 | 113 | 145 | **211** | 82 | 93 |
| Memory for the whole server | ~98 GB | ~97 GB | ~103 GB | **~33 GB** | ~101 GB | ~91 GB |

Nanbeige is good with tools. BFCL single-turn is 84.6 and multi-turn 56.5, both a little above Qwen FP4-Local, the best model so far for a single Spark. The gap is within the noise, so call it a tie with the best. On GPQA-Diamond it ties Qwen at 64.6. Ten of its GPQA answers ran into the 4,096-token limit, all scored wrong, so with more room it would likely end a few points higher.

Prompt injection is where it falls apart. 83 of 200 injected instructions got through, against 2 for Qwen FP4-Local: 26 of 29 in Slack, 22 of 35 in travel, 14 of 23 in banking, 21 of 113 in the mail and calendar suite. I checked a few transcripts: real transfers to the attacker's IBAN, and scheduled payments redirected to it. Under attack it finishes only 33% of the tasks cleanly (task done, attack not carried out), against 81% for Qwen.

The rest is weaker too. IFEval is 74.7, the lowest in the table, and only 12 of its failures ran into the token limit. Dutch knowledge is 70.3, 17 points below Qwen. Like Flint, it thinks the capital of West Flanders is Kortrijk.

And it is slow, because it is dense. Every token reads all 29 GB of weights, so decode tops out around 7 tok/s without help. The 3B drafter brings that to 12.3, and in the agent loop each step waits 6-8 seconds for the first token: 5.7 steps per minute, against 13.8 for Qwen FP4-Local. Prefill runs at a fifth of Qwen's speed (the kit's long prompt is 41K tokens with this tokenizer).

So Qwen FP4-Local stays the model for a single Spark. Nanbeige's tool calling is the real thing, but an agent that reads mail, chat or web pages needs to resist what's in them, and at 12 tok/s it's a slow way to find that out.

## Credits

- [Nanbeige](https://huggingface.co/Nanbeige) for Nanbeige4.3-33B and Nanbeige4.1-3B (Apache 2.0).
- [bartowski](https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF) for the imatrix GGUFs, [mradermacher](https://huggingface.co/mradermacher/Nanbeige4.1-3B-GGUF) for the 3B GGUF. The weights are not in this repository.
- The shape of this repository follows [Underdog-Saluki-27B-DGX-Spark](https://github.com/dedene/Underdog-Saluki-27B-DGX-Spark): one Compose project, pinned downloads, and a small check.

## License

MIT, see [LICENSE](LICENSE). The model weights are Apache 2.0 by their authors.
