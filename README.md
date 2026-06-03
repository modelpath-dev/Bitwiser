# BitWiser

One-click 1-bit (IQ1_S) compressor for any local LLM on Apple Silicon. FastAPI backend, React frontend, llama.cpp + Metal under the hood.

## Screenshots

**Pick a model** - auto-discovered from Ollama and `~/.models`.

![Model picker](images/img1.png)

**Shrink to IQ1_S** - importance-matrix calibration, Metal-tuned quantize, live RAM / bandwidth / file-size readout.

![Compression](images/img2.png)

**Side-by-side playground** - same prompt, original vs 1-bit, with tokens/sec and elapsed time.

![Playground](images/img3.png)

## Install

```bash
brew install cmake
./setup.sh                                  # builds llama.cpp with Metal + Python deps
cd web && npm install                       # frontend deps
```

`setup.sh` produces `vendor/llama.cpp/build/bin/llama-quantize` and `llama-cli`. ~10 minutes the first time.

## Run (dev)

Two terminals:

```bash
# terminal 1 - API
.venv/bin/python server.py                  # http://127.0.0.1:8000

# terminal 2 - UI
cd web && npm run dev                       # http://localhost:5173
```

Vite proxies `/api/*` to the backend, so the UI just calls relative URLs.

## TL;DR — results
 
Hardware: **Apple M3 Max (40-core GPU), 64 GB** · `llama.cpp` build `b4327` · perplexity on **WikiText-2 test** (~245K tokens, ctx 512).
 
| Model | Precision | Size (GB) | Compression | PPL (WikiText-2) | ΔPPL vs FP16 | Tok/s (decode) | Peak RAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | FP16 | 16.1 | 1.0× | 6.85 | — | 14.2 | 16.4 |
| Llama-3.1-8B | Q4_K_M *(ref)* | 4.92 | 3.3× | 7.04 | +0.19 | 58.7 | 5.9 |
| Llama-3.1-8B | IQ1_S *(no imatrix)* | 2.19 | 7.4× | 28.41 | +21.56 | 51.3 | 3.1 |
| Llama-3.1-8B | **IQ1_S (imatrix)** | **2.19** | **7.4×** | **10.18** | **+3.33** | **51.0** | **3.1** |
| Mistral-7B | FP16 | 14.48 | 1.0× | 5.94 | — | 16.1 | 14.8 |
| Mistral-7B | **IQ1_S (imatrix)** | **1.98** | **7.3×** | **9.47** | **+3.53** | **54.2** | **2.8** |
 
**Headline:** IQ1_S with importance-matrix calibration compresses Llama-3.1-8B from `16.1 GB → 2.19 GB` (`7.4×`) for a perplexity cost of `+3.33` (`+49%`), running at `~51 tok/s` on `3.1 GB` of RAM — fitting on hardware where FP16 barely runs. The cost is real, not "minimal"; the win is that the model fits and stays coherent at all.
 
---

## Run (single-process)

```bash
cd web && npm run build                     # emits web/dist/
.venv/bin/python server.py                  # serves the built UI from /
```

## CLI

```bash
.venv/bin/python scanner.py                 # JSON list of discovered models
.venv/bin/python compressor.py "library/gemma4:latest"
```

## How it works
 
1. **Discover** — FastAPI backend auto-detects Ollama models and on-disk GGUFs.
2. **Calibrate** — computes an importance matrix from a calibration corpus (`llama-imatrix`) so the quantizer knows which weights to protect.
3. **Quantize** — one call to `llama-quantize ... IQ1_S` using the imatrix; outputs a GGUF ~7× smaller.
4. **Benchmark** — React playground loads original + 1-bit side by side and streams live **tokens/sec, peak RAM, and memory bandwidth** while you prompt both.
```
on-disk / Ollama models
        │  (auto-discover)
        ▼
  imatrix calibration  ──►  IQ1_S quantize (Metal)  ──►  GGUF (~7× smaller)
        │                                                     │
        └──────────────► FastAPI ◄────────────────────────────┘
                            │
                     React playground
              (orig vs 1-bit, live tok/s · RAM · bandwidth)
```
 
---

| File | Responsibility |
| --- | --- |
| `scanner.py` | Walks Ollama manifests + local `.gguf`, peeks GGUF headers for arch/quant. |
| `compressor.py` | Wraps `llama-quantize`, streams progress, caches output to `compressed/`. |
| `prompts.py` | Per-family chat templates (Gemma / Llama / Mistral / Phi / Qwen). |
| `inference.py` | Spawns `llama-cli` with optimal threads, `-ngl 99`, mmap fallback for OOM. |
| `monitor.py` | psutil RSS/CPU sampler + optional `powermetrics` wattage. |
| `server.py` | FastAPI: `/api/models`, `/api/compress` (SSE), `/api/inference` (SSE). |
| `web/` | Vite + React + Tailwind v4 frontend. |

## Notes

- **Memory fallback** - if a model is bigger than free RAM, `inference.py` warns the UI and runs with `mmap` so the kernel can page weights from disk.
- **Wattage** - `powermetrics` requires sudo. Configure passwordless sudo (`sudo visudo`) for live energy numbers; otherwise the dashboard estimates from bandwidth.
- **Threads** - runner picks `physical_cores - 2` and offloads all layers to Metal with `-ngl 99`.

## References

The idea that large language models can be aggressively compressed down to roughly 1 bit per weight without catastrophic quality loss comes from:

> **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits**
> Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research), 2024.
> [arxiv.org/pdf/2402.17764](https://arxiv.org/pdf/2402.17764)

That paper introduces **BitNet b1.58**, a Transformer variant whose weights are constrained to the ternary set {-1, 0, +1} (≈1.58 bits/weight) and still matches full-precision baselines on perplexity and downstream tasks while dramatically reducing memory, bandwidth, and energy. BitWiser is a practical, post-hoc analogue: instead of training a model in 1.58-bit form, it takes any already-trained GGUF and uses llama.cpp's **IQ1_S** (≈1.56 bits/weight, importance-matrix guided) quantization to hit a similar regime on a laptop. Quality is not preserved as well as in BitNet b1.58 — that's the price of compressing after the fact rather than training natively — but the speed and RAM wins translate directly.

Credit for the underlying inference and quantization tooling goes to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and its contributors, particularly the authors of the IQ-quant family (Kawrakow et al.).
