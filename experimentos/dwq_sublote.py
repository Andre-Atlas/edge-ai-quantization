"""DWQ do mlx-lm 0.32.2 com os alvos do professor calculados em SUB-LOTES de 1 amostra.

  python experimentos/dwq_sublote.py [--piso-gb 3] <argumentos do mlx_lm dwq, p.ex.: --model M --mlx-path D --bits 4
        --group-size 64 --seed 0 --grad-checkpoint --target-dir DIR>
  python experimentos/dwq_sublote.py --verificar [--model Qwen/Qwen3-0.6B]    # equivalencia numerica dos alvos, sem treinar

Motivo: em `mlx_lm.quant.dwq.compute_dwq_targets` o pico depende de lote x tokens x vocabulario (logits do lote inteiro e
`argpartition` sobre o vocabulario de ~152 mil tokens): 12,4 GB de MLX (14 a 20 GB de footprint) ja no Qwen3-0,6B. Aqui cada
amostra e processada sozinha e os resultados sao concatenados no mesmo arquivo por lote (mesma indexacao, que o treino exige).
As linhas sao independentes, entao os alvos sao os MESMOS a menos de ruido numerico do bf16 (--verificar mede isso). Ainda
valem as ressalvas do --target-dir: os alvos guardam so os 1024 maiores logits, o que NAO e identico ao DWQ padrao (logits
completos); a comparacao de qualidade contra o padrao e uma etapa a parte (comparar_gptq.py).
Trava de seguranca: aborta se a memoria disponivel cair abaixo de --piso-gb.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import psutil
from tqdm import tqdm

import mlx_lm.quant.dwq as dwq
from mlx_lm.tuner.trainer import iterate_batches

GB = 1e9


def alvos_do_lote(model, batch, k=1024):
    """Logits e indices dos k maiores logits, sub-lote a sub-lote (igual ao compute_dwq_targets, linha a linha)."""
    vals, ids = [], []
    for j in range(batch.shape[0]):
        lg = model(batch[j : j + 1])
        lg = mx.stop_gradient(lg, stream=mx.cpu)   # mesmo truque do original (ultima operacao na CPU)
        mx.eval(lg)
        idx = mx.argpartition(lg, kth=-k, axis=-1)[..., -k:]
        vals.append(mx.take_along_axis(lg, idx, axis=-1))
        ids.append(idx)
        mx.eval(vals[-1], ids[-1])
        del lg
        mx.clear_cache()
    return mx.concatenate(vals, axis=0), mx.concatenate(ids, axis=0)


def compute_dwq_targets_sublote(model, save_dir, train_data, valid_data, batch_size, max_seq_length, seed):
    rank = mx.distributed.init().rank()

    def _compute_targets(data, path, split):
        if rank == 0:
            path = path / split
            path.mkdir(parents=True, exist_ok=True)
        for i, (batch, _) in tqdm(
            enumerate(iterate_batches(data, batch_size, max_seq_length, seed=seed)),
            total=len(data) // batch_size, desc=f"Computing targets for {split} (sub-lotes)", disable=rank != 0,
        ):
            batch = batch[:, :-1]
            logits, idx = alvos_do_lote(model, batch)
            if rank == 0:
                mx.save_safetensors(path / f"{i:010d}.safetensors", {"logits": logits, "indices": idx})

    _compute_targets(valid_data, save_dir, "valid")
    _compute_targets(train_data, save_dir, "train")


def verificar(modelo):
    """Compara o calculo em lote (original) com o em sub-lotes num lote aleatorio: indices e valores."""
    from mlx_lm.utils import load

    model, _ = load(modelo)
    mx.random.seed(0)
    batch = mx.random.randint(0, 1000, (4, 256))
    lg = model(batch)
    idx = mx.argpartition(lg, kth=-1024, axis=-1)[..., -1024:]
    ref_v = mx.take_along_axis(lg, idx, axis=-1)
    mx.eval(ref_v, idx)
    v, i = alvos_do_lote(model, batch)
    a = mx.sort(ref_v.astype(mx.float32), axis=-1)
    b = mx.sort(v.astype(mx.float32), axis=-1)
    mesmo_conj = float(mx.mean((mx.sort(idx, axis=-1) == mx.sort(i, axis=-1)).astype(mx.float32)).item())
    print(f"[verificacao] indices iguais (como conjuntos, por posicao): {100 * mesmo_conj:.3f}%; "
          f"max|dif| dos logits ordenados: {float(mx.abs(a - b).max().item()):.4g} (escala dos logits: "
          f"{float(mx.abs(a).max().item()):.1f})", flush=True)


def vigiar(piso_gb):
    def f():
        while True:
            if psutil.virtual_memory().available < piso_gb * GB:
                print(f"\n[ABORTADO] memoria disponivel < {piso_gb} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=f, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--piso-gb", type=float, default=3.0)
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    a, resto = ap.parse_known_args()
    vigiar(a.piso_gb)
    if a.verificar:
        return verificar(a.model)
    dwq.compute_dwq_targets = compute_dwq_targets_sublote      # main() chama a funcao pelo nome do modulo
    sys.argv = ["mlx_lm dwq", "--model", a.model, *resto]
    t0 = time.time()
    dwq.main()
    print(f"[dwq_sublote] pico MLX {mx.get_peak_memory() / GB:.2f} GB; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
