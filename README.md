# tinyinfer: From-Scratch Transformer Inference Engine in Pure NumPy

`tinyinfer` is a modular, high-performance Transformer inference runtime implemented completely from first principles in pure **NumPy**. It executes autoregressive inference for generative language models (GPT-2 architecture) on CPU with **zero deep learning frameworks (no PyTorch, no TensorFlow, no HuggingFace)** in the inference path.

---

## Architecture Overview

```mermaid
graph TD
    subgraph Input_Pipeline ["Input & Embedding Pipeline"]
        Tokens["Token IDs (B, T)"] --> TokenEmb["Token Embeddings wte (B, T, C)"]
        Pos["Position IDs [p..p+T]"] --> PosEmb["Positional Embeddings wpe (T, C)"]
        TokenEmb & PosEmb --> SumEmb["Input Representation x = wte + wpe (B, T, C)"]
    end

    subgraph Transformer_Layer ["Transformer Decoder Block (x 12 Layers)"]
        SumEmb --> LN1["Pre-LayerNorm 1"]
        LN1 --> QKV["Fused QKV Linear Projection (B, T, 3*C)"]
        QKV --> Split["Split into Q, K, V & Multi-Head Reshape (B, nh, T, hs)"]
        
        Split -->|K_new, V_new| KVCache["KV-Cache: Past Key/Value Buffer (B, nh, past_len+T, hs)"]
        Split -->|Q| AttnScores["Scaled Dot-Product Attention: Q @ K.T / sqrt(hs)"]
        KVCache --> AttnScores
        
        AttnScores --> Mask["Causal Masking (Lower Triangular for Prefill / No-op for Decode)"]
        Mask --> Softmax["Softmax over Key Dimension"]
        Softmax & KVCache --> Context["Value Aggregation: Attn @ V"]
        Context --> OutProj["Output Projection c_proj (B, T, C)"]
        
        OutProj & SumEmb --> Residual1["Residual Addition 1: x + Attn"]
        Residual1 --> LN2["Pre-LayerNorm 2"]
        LN2 --> FC["MLP Expansion: c_fc (C -> 4*C)"]
        FC --> GELU["GELU Activation (Tanh Approximation)"]
        GELU --> MLPProj["MLP Contraction: c_proj (4*C -> C)"]
        MLPProj & Residual1 --> Residual2["Residual Addition 2: x + MLP"]
    end

    subgraph Output_Pipeline ["Language Modeling Head"]
        Residual2 --> LNFinal["Final LayerNorm ln_f"]
        LNFinal --> LMHead["Vocabulary Logits Projection: x @ lm_head.T (B, T, vocab_size)"]
        LMHead --> Sampling["Sampling Engine (Greedy Argmax / Temperature / Top-k)"]
        Sampling --> NextToken["Next Token ID"]
    end
```

---

## Core Milestones & Staged Implementations

### Stage 1: Pure NumPy GPT-2 Engine & Numerical Parity
- **Weight Loader (`weights.py`)**: Direct extraction from HuggingFace `model.safetensors` binary format into structured NumPy arrays without PyTorch dependencies.
- **Offline Tokenizer (`tokenizer.py`)**: Self-contained BPE tokenizer loading `tokenizer.json`, `vocab.json`, and `merges.txt`.
- **Pure NumPy Math Kernels (`tensor_ops.py`)**:
  - `layer_norm`: Mean, variance, epsilon stabilization, gain/bias scaling.
  - `gelu`: Exact tanh approximation $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$.
  - `softmax`: Numerically stable with max subtraction across arbitrary axes.
  - `linear`: Matrix multiplication supporting both standard and Conv1D weight orientations.
- **Verification (`test_stage1_forward.py`)**: Evaluated across 5 diverse prompts against HuggingFace `GPT2LMHeadModel`. **100% next-token identity** with mean absolute logit difference $\approx 2 \times 10^{-5}$ ($< 2 \times 10^{-4}$ maximum tail difference across all 50,257 vocabulary logits).

---

### Stage 2: Key-Value (KV) Cache for Incremental Decoding
During naive autoregressive decoding, generating token $t+1$ requires passing the full sequence $1..t$ through all 12 layers. This recomputes past keys and values unnecessarily, resulting in $O(N^2)$ cumulative computational complexity.

`tinyinfer` implements an incremental **KV-Cache** (`kv_cache.py`):
1. **Prefill Phase**: The initial prompt sequence $T_{\text{prompt}}$ is evaluated in a single forward pass, populating the cache with keys $K_0$ and values $V_0$ of shape `(B, n_head, T_prompt, head_dim)`.
2. **Decode Phase**: Each subsequent generation step passes **only 1 new token** ($T_{\text{new}} = 1$) with position index $p = \text{past\_len}$.
3. **Causal Mask Optimization**: In single-token decoding, every cached key corresponds to a past position $j \le p$. Consequently, all attention weights are causal by definition, completely bypassing mask computation.
4. **Rollback & Truncate**: Built-in `cache.truncate(new_length)` enabling instant cache rollback for speculative decoding.

#### Measured Benchmark Results (Intel i5 CPU)
Benchmarked using `tinyinfer/bench_kv_cache.py`:

| Generated Tokens ($N$) | Latency (With Cache) | Speed (With Cache) | Latency (Without Cache) | Speed (Without Cache) | **Speedup** | Peak Cache RAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16 tokens** | 23.25 s | 0.69 tok/s | 34.15 s | 0.47 tok/s | **1.47x** | 2.96 MB |
| **32 tokens** | 50.18 s | 0.64 tok/s | 73.54 s | 0.44 tok/s | **1.47x** | 5.12 MB |
| **64 tokens** | 88.22 s | 0.73 tok/s | 182.47 s | 0.35 tok/s | **2.07x** | 9.43 MB |

*The speedup scales super-linearly with sequence length as $O(N)$ cached generation outpaces $O(N^2)$ full recomputation.*

---

### Stage 3: INT8 Per-Channel Weight Quantization
To compress memory bandwidth and footprint for edge deployment, `tinyinfer` implements asymmetric (affine) per-channel weight quantization (`quantize.py`):

$$\text{Scale}_c = \frac{W_{\max, c} - W_{\min, c}}{255.0}, \quad \text{ZeroPoint}_c = \text{round}\left(-\frac{W_{\min, c}}{\text{Scale}_c}\right)$$

$$W_{\text{int8}} = \text{clip}\left(\text{round}\left(\frac{W_{\text{fp32}}}{\text{Scale}_c}\right) + \text{ZeroPoint}_c, 0, 255\right)$$

$$\hat{W}_{\text{fp32}} = \text{Scale}_c \odot (W_{\text{int8}} - \text{ZeroPoint}_c)$$

#### Measured Quantization Results
Benchmarked using `tinyinfer/bench_quantize.py` on standard evaluation text:

| Representation | Weights Memory (MB) | Compression Ratio | Perplexity (PPL) | Perplexity Delta | Generation Fidelity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 621.94 MB | 1.00x | 23.1729 | Baseline (0.00%) | 100% Reference |
| **INT8 (Per-Channel)** | 158.73 MB | **3.92x** | 23.2574 | **+0.36%** | Token-for-token identical text |

- **Weight Footprint**: Reduced from **621.94 MB down to 158.73 MB** (**3.92x compression**).
- **Quality Retention**: Perplexity degradation is only **+0.36%**, preserving the linguistic accuracy and semantic coherence of the original FP32 model.

---

### Stage 4: Lossless Speculative Decoding (`speculative.py`)
`tinyinfer` implements exact greedy speculative decoding pairing a fast draft model (such as `QuantizedGPT2Weights` INT8) with the authoritative FP32 target model:
1. **Draft Phase**: Proposes $\gamma$ candidate tokens autoregressively using the draft KV-cache.
2. **Batched Verification**: Evaluates all $\gamma$ candidates in a single target forward pass (`gpt2_forward`).
3. **KV-Cache Rollback**: On the first mismatch, rolls back both target and draft caches cleanly via `KVCache.truncate(keep_len)` while emitting the corrected token—preserving **100% token-for-token parity** with standard greedy decoding.

---

## Project Structure

```
tinyinfer/
├── attention.py          # Multi-head causal self-attention with KV-cache integration
├── bench_kv_cache.py     # KV-cache latency & throughput benchmarking suite
├── bench_quantize.py     # FP32 vs INT8 perplexity, speed & memory benchmark
├── generate.py           # Autoregressive generation loop with metrics tracking
├── kv_cache.py           # KVCache data structure with append, truncate, and memory stats
├── model.py              # TransformerBlock and complete GPT-2 forward pass
├── quantize.py           # Affine per-channel INT8 quantization and perplexity scoring
├── sampling.py           # Greedy argmax, temperature scaling, top-k sampling
├── tensor_ops.py         # Pure NumPy LayerNorm, GELU, Softmax, Linear kernels
├── tokenizer.py          # Offline BPE tokenizer loader
├── weights.py            # Direct safetensors weight parser
├── models/gpt2/          # Weights, config, and tokenizer files
└── tests/
    ├── test_stage1_forward.py   # Parity validation against PyTorch/HuggingFace
    ├── test_stage2_kv.py        # KV-cache correctness & logit invariance tests
    ├── test_stage3_quantize.py  # INT8 reconstruction error & perplexity tests
    └── test_stage4_speculative.py # Speculative decoding parity & rollback tests
```

---

## Quickstart & Execution Commands

### 0. Download GPT-2 Weights
```bash
python download_weights.py
```

### 1. Interactive Text Generation Demo
Run generation with the pure NumPy engine:
```bash
python -c "from tinyinfer.weights import load_gpt2_weights; from tinyinfer.tokenizer import LocalGPT2Tokenizer; from tinyinfer.generate import generate; w = load_gpt2_weights('tinyinfer/models/gpt2'); tok = LocalGPT2Tokenizer('tinyinfer/models/gpt2'); print(generate('Alan Turing was', w, tok, max_new_tokens=20, use_cache=True))"
```

### 2. Run All Automated Test Suites
Run comprehensive verification across all stages:
```bash
# Stage 1: Numerical parity with HuggingFace (5 prompts)
pytest tinyinfer/tests/test_stage1_forward.py -v -s

# Stage 2: KV-Cache step-by-step logit parity and end-to-end token match
pytest tinyinfer/tests/test_stage2_kv.py -v -s

# Stage 3: INT8 quantization reconstruction bounds, compression ratio & perplexity
pytest tinyinfer/tests/test_stage3_quantize.py -v -s

# Stage 4: Speculative decoding exact parity & KV-cache rollback
pytest tinyinfer/tests/test_stage4_speculative.py -v -s
```

### 3. Run Benchmarks
Generate the benchmark tables and plots:
```bash
# Benchmark KV-Cache throughput (generates results_kv_cache.csv and results_kv_cache.png)
python tinyinfer/bench_kv_cache.py --tokens 16 32 64

# Benchmark INT8 Quantization (generates results_quantize.csv)
python tinyinfer/bench_quantize.py --eval-tokens 16

# Benchmark Speculative Decoding (INT8 draft -> FP32 target)
python -m tinyinfer.bench_speculative --tokens 24 --gamma 4
```
