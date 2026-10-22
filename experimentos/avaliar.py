"""Avalia um modelo MLX: perplexidade no WikiText-2 (teste) e velocidade/memoria.

Uso: python experimentos/avaliar.py <caminho_ou_id_do_modelo> [--seq 2048]
Imprime um JSON com as metricas.
"""
import argparse
import json
import math
import time

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load, stream_generate


def ppl_wikitext2(model, tokenizer, seq_len=2048, batch_size=4):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texto = "\n\n".join(ds["text"])
    ids = mx.array(tokenizer.encode(texto))
    n = (ids.size // seq_len) * seq_len
    blocos = ids[:n].reshape(-1, seq_len)
    nll, ntok = 0.0, 0
    for i in range(0, blocos.shape[0], batch_size):
        x = blocos[i : i + batch_size]
        logits = model(x[:, :-1]).astype(mx.float32)
        loss = mx.sum(mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, x[:, 1:, None], axis=-1).squeeze(-1))
        mx.eval(loss)
        nll += loss.item()
        ntok += x[:, 1:].size
    return math.exp(nll / ntok), ntok


def velocidade(model, tokenizer, prompt_tokens=512, gen_tokens=128, trials=5):
    prompt = mx.array(tokenizer.encode("hello " * prompt_tokens)[:prompt_tokens])
    pre, dec = [], []
    for t in range(trials + 1):  # a 1a rodada e aquecimento
        r = None
        for r in stream_generate(model, tokenizer, prompt, max_tokens=gen_tokens):
            pass
        if t > 0:
            pre.append(r.prompt_tps)
            dec.append(r.generation_tps)
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("--seq", type=int, default=2048)
    a = ap.parse_args()
    model, tok = load(a.modelo)
    mx.reset_peak_memory()
    ppl, ntok = ppl_wikitext2(model, tok, a.seq)
    mx.reset_peak_memory()
    pre, dec = velocidade(model, tok)
    media = lambda v: sum(v) / len(v)
    dp = lambda v: (sum((x - media(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    print(json.dumps({
        "ppl_wikitext2": ppl, "tokens_avaliados": ntok, "seq": a.seq,
        "prefill_tps": media(pre), "prefill_tps_dp": dp(pre),
        "decode_tps": media(dec), "decode_tps_dp": dp(dec),
        "pico_memoria_gb": mx.get_peak_memory() / 1e9,
    }))


if __name__ == "__main__":
    main()
