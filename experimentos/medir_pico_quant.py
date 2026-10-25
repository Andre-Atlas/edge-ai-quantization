"""Mede o pico de memoria da QUANTIZACAO (gptq/awq/dwq), que o resultados.csv nao registra.

  python experimentos/medir_pico_quant.py --modelo Qwen/Qwen3-1.7B --metodos gptq,awq,dwq [--bits 4]

Cada metodo roda num processo proprio (pico limpo), com o mesmo comando de rodar.quantizar, mas dentro de um
wrapper que le mx.get_peak_memory() ao fim e amostra a memoria do sistema (psutil) a cada 0,5 s. Grava em
resultados/pico-quantizacao.csv e compara com matriz.estimativa_gb. Rode com a maquina sem outros jobs.
Trava de seguranca: se a memoria disponivel do sistema cair abaixo de --piso-gb, o filho e encerrado.
"""
import argparse
import csv
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "resultados" / "pico-quantizacao.csv"
FOOTPRINT = RAIZ / "resultados" / "pico-footprint.csv"
GB = 1e9


def filho(metodo, modelo, bits, grupo, seed, destino, resumo, piso_gb, extra=""):
    import mlx.core as mx
    import psutil

    st = dict(min_disp=psutil.virtual_memory().available, swap0=psutil.swap_memory().used, swap_max=0, abortado=False)

    def amostrar():
        while True:
            st["min_disp"] = min(st["min_disp"], psutil.virtual_memory().available)
            st["swap_max"] = max(st["swap_max"], psutil.swap_memory().used)
            if st["min_disp"] < piso_gb * GB:
                st["abortado"] = True
                Path(resumo).write_text(json.dumps(_fechar(st, mx, 0, True)))
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=amostrar, daemon=True).start()
    if metodo in ("fp", "rtn"):   # mesma linha de comando de rodar.quantizar
        argv = ["mlx_lm", "convert", "--hf-path", modelo, "--mlx-path", destino, "--dtype", "bfloat16"]
        if metodo == "rtn":
            argv += ["-q", "--q-bits", str(bits), "--q-group-size", str(grupo)]
    else:
        argv = ["mlx_lm", metodo, "--model", modelo, "--mlx-path", destino, "--bits", str(bits),
                "--group-size", str(grupo), "--seed", str(seed)]
    if metodo == "dwq":
        argv.append("--grad-checkpoint")
    argv += shlex.split(extra)
    sys.argv = argv
    t0 = time.time()
    try:
        runpy.run_module("mlx_lm", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    Path(resumo).write_text(json.dumps(_fechar(st, mx, time.time() - t0, False)))


def _fechar(st, mx, seg, abortado):
    return dict(pico_mlx_gb=round(mx.get_peak_memory() / GB, 2), min_disponivel_gb=round(st["min_disp"] / GB, 2),
                swap_extra_gb=round((st["swap_max"] - st["swap0"]) / GB, 2), segundos=round(seg), abortado=abortado)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--metodos", default="gptq,awq,dwq")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--grupo", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piso-gb", type=float, default=1.5)
    ap.add_argument("--saida", help="mantem o modelo quantizado nesta pasta (padrao: pasta temporaria apagada)")
    ap.add_argument("--extra", default="", help="argumentos extras da ferramenta (ex.: \"--target-dir DIR\" no dwq)")
    ap.add_argument("--filho", help=argparse.SUPPRESS)
    ap.add_argument("--destino", help=argparse.SUPPRESS)
    ap.add_argument("--resumo", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.filho:
        return filho(a.filho, a.modelo, a.bits, a.grupo, a.seed, a.destino, a.resumo, a.piso_gb, a.extra)

    sys.path.insert(0, str(RAIZ))
    import matriz

    cfg = json.load(open(RAIZ / "matriz.json"))
    mod = next(m for m in cfg["modelos"] if m["id"] == a.modelo)
    fp_gb, arq = mod["fp_gb"], mod.get("arq")
    SAIDA.parent.mkdir(exist_ok=True)
    novo = not SAIDA.exists()
    with open(SAIDA, "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["data", "modelo", "metodo", "bits", "fp_gb", "estimado_gb", "pico_mlx_gb", "min_disponivel_gb",
                        "swap_extra_gb", "segundos", "abortado"])
        for met in a.metodos.split(","):
            est = matriz.estimativa_gb(dict(metodo=met, fp_gb=fp_gb, bits=a.bits, arq=arq))
            with tempfile.TemporaryDirectory() as tmp:
                resumo = str(Path(tmp) / "r.json")
                cmd = [sys.executable, __file__, "--filho", met, "--modelo", a.modelo, "--bits", str(a.bits),
                       "--grupo", str(a.grupo), "--seed", str(a.seed), "--destino", a.saida or str(Path(tmp) / "m"), "--extra=" + a.extra,
                       "--resumo", resumo, "--piso-gb", str(a.piso_gb)]
                print(f"[medindo] {a.modelo} {met} {a.bits}b (estimado {est:.1f} GB)", flush=True)
                # /usr/bin/time -l acrescenta o footprint do processo (o que de fato pressiona o sistema; inclui o que o MLX nao conta)
                p = subprocess.run(["/usr/bin/time", "-l", *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                fm = re.search(r"(\d+)\s+peak memory footprint", p.stderr or "")
                footprint = round(int(fm.group(1)) / GB, 2) if fm else None
                if not Path(resumo).exists():
                    print(f"[FALHOU] {met} (codigo {p.returncode})", flush=True)
                    continue
                r = json.load(open(resumo))
            w.writerow([time.strftime("%Y-%m-%d"), a.modelo, met, a.bits, fp_gb, round(est, 1), r["pico_mlx_gb"],
                        r["min_disponivel_gb"], r["swap_extra_gb"], r["segundos"], r["abortado"]])
            f.flush()
            novo_fp = not FOOTPRINT.exists()
            with open(FOOTPRINT, "a", newline="") as g:
                wf = csv.writer(g)
                if novo_fp:
                    wf.writerow(["data", "modelo", "metodo", "bits", "fp_gb", "pico_mlx_gb", "footprint_gb", "abortado"])
                wf.writerow([time.strftime("%Y-%m-%d"), a.modelo, met, a.bits, fp_gb, r["pico_mlx_gb"], footprint, r["abortado"]])
            print(f"[footprint] {met}: {footprint} GB (MLX {r['pico_mlx_gb']} GB)", flush=True)
            print(f"[ok] {met}: pico MLX {r['pico_mlx_gb']} GB vs estimado {est:.1f} GB; min disponivel do sistema "
                  f"{r['min_disponivel_gb']} GB; swap extra {r['swap_extra_gb']} GB; {r['segundos']}s"
                  f"{'  ABORTADO' if r['abortado'] else ''}", flush=True)


if __name__ == "__main__":
    main()
