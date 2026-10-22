"""Quantiza (RTN/GPTQ/AWQ/DWQ) ou converte (fp) um modelo e acrescenta uma linha em resultados/resultados.csv.

Uso: python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo rtn --bits 4 [--grupo 64]
Metodos: fp (bf16, referencia), rtn, gptq, awq, dwq.
"""
import argparse
import csv
import datetime
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
MODELOS = RAIZ / "modelos"
CSV = RAIZ / "resultados" / "resultados.csv"
PY = sys.executable


def quantizar(modelo, metodo, bits, grupo, destino):
    base = [PY, "-m", "mlx_lm"]
    if metodo == "fp":
        cmd = base + ["convert", "--hf-path", modelo, "--mlx-path", str(destino), "--dtype", "bfloat16"]
    elif metodo == "rtn":
        cmd = base + ["convert", "--hf-path", modelo, "--mlx-path", str(destino), "-q",
                      "--q-bits", str(bits), "--q-group-size", str(grupo), "--dtype", "bfloat16"]
    elif metodo in ("gptq", "awq"):
        cmd = base + [metodo, "--model", modelo, "--mlx-path", str(destino),
                      "--bits", str(bits), "--group-size", str(grupo)]
    elif metodo == "dwq":
        cmd = base + ["dwq", "--model", modelo, "--mlx-path", str(destino),
                      "--bits", str(bits), "--group-size", str(grupo)]
    else:
        raise SystemExit(f"metodo desconhecido: {metodo}")
    t0 = datetime.datetime.now()
    subprocess.run(cmd, check=True)
    return (datetime.datetime.now() - t0).total_seconds()


def tamanho_gb(pasta):
    return sum(f.stat().st_size for f in Path(pasta).rglob("*") if f.is_file()) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--metodo", required=True, choices=["fp", "rtn", "gptq", "awq", "dwq"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--grupo", type=int, default=64)
    ap.add_argument("--manter", action="store_true", help="nao apagar o modelo quantizado")
    a = ap.parse_args()

    nome = f"{a.modelo.split('/')[-1]}-{a.metodo}" + ("" if a.metodo == "fp" else f"-{a.bits}b-g{a.grupo}")
    destino = MODELOS / nome
    if destino.exists():
        shutil.rmtree(destino)
    MODELOS.mkdir(exist_ok=True)
    t_quant = quantizar(a.modelo, a.metodo, a.bits, a.grupo, destino)

    saida = subprocess.run([PY, str(RAIZ / "avaliar.py"), str(destino)], check=True,
                           capture_output=True, text=True).stdout.strip().splitlines()[-1]
    m = json.loads(saida)
    linha = {
        "data": datetime.date.today().isoformat(), "modelo": a.modelo, "metodo": a.metodo,
        "bits": 16 if a.metodo == "fp" else a.bits, "grupo": "" if a.metodo == "fp" else a.grupo,
        "tamanho_gb": round(tamanho_gb(destino), 3), "tempo_quant_s": round(t_quant, 1),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
        "mlx_lm": md.version("mlx-lm"), "mlx": md.version("mlx"), "python": platform.python_version(),
    }
    CSV.parent.mkdir(exist_ok=True)
    novo = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linha))
        if novo:
            w.writeheader()
        w.writerow(linha)
    print(json.dumps(linha, ensure_ascii=False, indent=1))
    if not a.manter:
        shutil.rmtree(destino)


if __name__ == "__main__":
    main()
