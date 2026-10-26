"""Fase de velocidade dos K-quants (llama.cpp): prefill e decodificacao de cada GGUF, com a maquina ociosa.

  python experimentos/kquant_velocidade.py [--modelos Qwen3-0.6B,...] [--tipos F16,Q8_0,...] [--csv ARQ] [--refazer]

Mesmos parametros do MLX (avaliar.velocidade) e do comando documentado no topo do kquant.py: prompt de 512 tokens,
128 gerados, 5 repeticoes, todas as camadas na GPU (`llama-bench -p 512 -n 128 -r 5 -ngl 99 -o csv`). Grava uma linha
por modelo x tipo em resultados/kquant-velocidade.csv (retomavel: pula o que ja existe). Os GGUF vem de
experimentos/ferramentas/gguf/<modelo>/<modelo>-<TIPO>.gguf (gerados pelo kquant.py). Rodar SOZINHO na maquina.
"""
import argparse
import csv
import datetime
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kquant import BIN, RAIZ, versao  # noqa: E402

GGUF = RAIZ / "experimentos/ferramentas/gguf"
SAIDA = RAIZ / "experimentos/resultados/kquant-velocidade.csv"
TIPOS = ["F16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"]
CAMPOS = ["modelo", "tipo", "prefill_tps", "prefill_tps_dp", "decode_tps", "decode_tps_dp", "tamanho_gb",
          "n_prompt", "n_gen", "repeticoes", "llamacpp_versao", "data"]


def medir(gguf):
    """Roda o llama-bench e devolve (pp512 media, dp, tg128 media, dp) em tokens/s."""
    r = subprocess.run([str(BIN / "llama-bench"), "-m", str(gguf), "-p", "512", "-n", "128", "-r", "5", "-ngl", "99",
                        "-o", "csv"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    linhas = list(csv.DictReader(io.StringIO(r.stdout)))
    pp = next(x for x in linhas if int(x["n_prompt"]) > 0 and int(x["n_gen"]) == 0)
    tg = next(x for x in linhas if int(x["n_gen"]) > 0 and int(x["n_prompt"]) == 0)
    return float(pp["avg_ts"]), float(pp["stddev_ts"]), float(tg["avg_ts"]), float(tg["stddev_ts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", help="nomes curtos separados por virgula (padrao: todos em ferramentas/gguf)")
    ap.add_argument("--tipos", default=",".join(TIPOS))
    ap.add_argument("--csv", type=Path, default=SAIDA)
    ap.add_argument("--refazer", action="store_true")
    a = ap.parse_args()
    modelos = a.modelos.split(",") if a.modelos else sorted(p.name for p in GGUF.iterdir() if p.is_dir())
    feitos = set()
    if a.csv.exists() and not a.refazer:
        feitos = {(r["modelo"], r["tipo"]) for r in csv.DictReader(open(a.csv)) if r.get("decode_tps")}
    novo = not a.csv.exists()
    ver = versao()
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        for m in modelos:
            for t in a.tipos.split(","):
                gguf = GGUF / m / f"{m}-{t}.gguf"
                if (m, t) in feitos:
                    print(f"[ja feito] {m} {t}", flush=True)
                    continue
                if not gguf.exists():
                    print(f"[FALHOU] {m} {t}: GGUF ausente ({gguf.name})", flush=True)
                    continue
                try:
                    pp, ppd, tg, tgd = medir(gguf)
                except Exception as e:  # noqa: BLE001 - registra e segue para o proximo
                    print(f"[FALHOU] {m} {t}: {str(e)[-300:]}", flush=True)
                    continue
                w.writerow(dict(modelo=m, tipo=t, prefill_tps=round(pp, 2), prefill_tps_dp=round(ppd, 2),
                                decode_tps=round(tg, 2), decode_tps_dp=round(tgd, 2),
                                tamanho_gb=round(gguf.stat().st_size / 1e9, 3), n_prompt=512, n_gen=128, repeticoes=5,
                                llamacpp_versao=ver, data=datetime.date.today().isoformat()))
                f.flush()
                print(f"[ok] {m} {t}: prefill {pp:.1f} ± {ppd:.1f} tok/s, decode {tg:.1f} ± {tgd:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
