"""Perfil de memoria do AWQ do mlx-lm 0.32.2: onde o pico acontece e se a memoria se acumula de um bloco para o outro.

  python experimentos/perfil_awq.py [--piso-gb 3] <argumentos do mlx_lm awq, p.ex.: --model M --mlx-path D --bits 4
        --group-size 64 --seed 0 [--num-samples N --sequence-length L]>

Envolve as funcoes do modulo mlx_lm.quant.awq (run_layer, search_best_scale, search_best_clip, apply_scale, scale_block,
clip_block) e imprime, a cada chamada, a memoria ATIVA do MLX antes e depois e o pico acumulado. Nao altera nenhum calculo
(so le contadores; as funcoes ja avaliam seus resultados). Se a memoria ativa ao inicio de cada bloco CRESCE, ha retencao entre
blocos (evitavel sem mudar o resultado); se so o pico dentro do bloco e alto, o custo e inerente as amostras de calibracao.
O parametro de verificacao --resumo imprime, ao fim, o maior pico por funcao.
"""
import argparse
import os
import sys
import threading
import time
from collections import defaultdict

import mlx.core as mx
import psutil

import mlx_lm.quant.awq as awq

GB = 1e9
FUNCOES = ["run_layer", "search_best_scale", "search_best_clip", "apply_scale", "scale_block", "clip_block"]
chamadas, maior = [0], defaultdict(float)


def envolver(nome):
    f = getattr(awq, nome)

    def g(*a, **k):
        antes = mx.get_active_memory() / GB
        r = f(*a, **k)
        depois, pico, cache = mx.get_active_memory() / GB, mx.get_peak_memory() / GB, mx.get_cache_memory() / GB
        chamadas[0] += 1
        maior[nome] = max(maior[nome], depois)
        print(f"[awq_perfil] #{chamadas[0]:03d} {nome:<18} ativa {antes:6.2f} -> {depois:6.2f} GB | pico acumulado {pico:6.2f} GB | cache {cache:5.2f} GB", flush=True)
        return r

    setattr(awq, nome, g)


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
    a, resto = ap.parse_known_args()
    vigiar(a.piso_gb)
    for n in FUNCOES:
        envolver(n)
    sys.argv = ["mlx_lm awq", *resto]
    t0 = time.time()
    awq.main()
    print(f"[awq_perfil] FIM: pico MLX {mx.get_peak_memory() / GB:.2f} GB; {time.time() - t0:.0f} s; {chamadas[0]} chamadas", flush=True)
    for n, v in sorted(maior.items(), key=lambda x: -x[1]):
        print(f"[awq_perfil] maior memoria ativa apos {n}: {v:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
