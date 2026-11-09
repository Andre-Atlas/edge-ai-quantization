# Tabela 1 — Modelos Avaliados, Dimensões Arquiteturais e Linha de Base (FP16 / BF16)

| Modelo | Camadas ($n$) | $d_{\text{model}}$ | $d_{\text{ff}}$ | $d_{\text{proj}}$ | Tam. BF16 (GB) | PPL WikiText-2 (MLX) | Acc Zero-Shot Média | Decode BF16 MLX (tok/s) | PPL F16 (`llama.cpp`) | Decode F16 (`llama.cpp`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-0.6B` | 28 | 1024 | 3072 | 2048 | 1.204 | 20.97 | 0.5215 | 196.3 | 18.39 | 179.1 |
| `Qwen3-1.7B` | 28 | 2048 | 6144 | 2048 | 3.453 | 16.72 | 0.6133 | 78.6 | 14.90 | 77.4 |
| `SmolLM3-3B` | 36 | 2048 | 11008 | 2048 | 6.167 | 9.12 | 0.7057 | 45.3 | 8.11 | 44.7 |
| `Llama-3.2-3B` | 28 | 3072 | 8192 | 3072 | 6.443 | 7.82 | 0.6795 | 43.0 | 6.99 | 43.8 |
| `Qwen3-4B` | 36 | 2560 | 9728 | 4096 | 8.056 | 13.66 | 0.6837 | 33.8 | 11.98 | 34.9 |
