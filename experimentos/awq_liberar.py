"""AWQ do mlx-lm 0.32.2 com a memoria liberada ao fim de cada bloco. NAO muda nenhum calculo.

  python experimentos/awq_liberar.py [--piso-gb 2.5] [--cache-gb 1.0] [--perfil] <argumentos do mlx_lm awq, p.ex.:
        --model M --mlx-path D --bits 4 --group-size 64 --seed 0>

Diagnostico (perfil_awq.py, 25/10/2026, Qwen3-0,6B): a memoria ativa ao fim de cada bloco cresce ~1,4 GB por bloco, o tamanho das
entradas guardadas (input_feat) do bloco. O awq.py atribui `input_feat` aos modulos e nunca o apaga; elas so saem quando o
coletor de lixo do Python roda. Aqui `mx.clear_cache()` (chamado pelo awq.py ao fim de cada bloco) passa a rodar antes um
`gc.collect()`, e o cache de buffers do MLX e limitado (--cache-gb): o cache chegou a 11,5 GB no perfil e conta no footprint
do processo. Nenhuma das duas mudancas altera valores: o resultado deve ser identico ao do AWQ padrao (comparar_gptq.py).
--perfil imprime a memoria ativa/pico/cache a cada etapa, como perfil_awq.py.
Trava de seguranca: aborta se a memoria disponivel cair abaixo de --piso-gb.
"""
import argparse
import gc
import os
import sys
import threading
import time

import mlx.core as mx
import psutil

import mlx_lm.quant.awq as awq

GB = 1e9


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
    ap.add_argument("--piso-gb", type=float, default=2.5)
    ap.add_argument("--cache-gb", type=float, default=1.0)
    ap.add_argument("--perfil", action="store_true")
    a, resto = ap.parse_known_args()
    vigiar(a.piso_gb)
    mx.set_cache_limit(int(a.cache_gb * GB))
    original = mx.clear_cache

    def clear_cache_com_gc():
        gc.collect()
        original()

    mx.clear_cache = clear_cache_com_gc                  # o awq.py chama mx.clear_cache() ao fim de cada bloco
    if a.perfil:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import perfil_awq

        for n in perfil_awq.FUNCOES:
            perfil_awq.envolver(n)
    sys.argv = ["mlx_lm awq", *resto]
    t0 = time.time()
    awq.main()
    print(f"[awq_liberar] pico MLX {mx.get_peak_memory() / GB:.2f} GB; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
