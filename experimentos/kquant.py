"""Metodo `kquant` (llama.cpp): GGUF F16 -> imatrix -> llama-quantize -> PPL/KLD.

Uso (a partir da raiz do projeto):
    .venv/bin/python experimentos/kquant.py --modelo Qwen/Qwen3-0.6B \
        --tipos Q8_0,Q4_K_M,Q3_K_M,Q2_K

Etapas: (1) snapshot do HF (cache local); (2) convert_hf_to_gguf.py -> F16;
(3) llama-imatrix sobre ~/.cache/mlx-lm/calibration_v5.txt (mesmo texto do mlx-lm);
(4) llama-quantize --imatrix para cada tipo; (5) llama-perplexity no WikiText-2 raw
teste ("\\n\\n".join do split, como em avaliar.py), janelas de 2048 sem sobreposicao;
o F16 grava os logits-base (--kl-divergence-base) e cada tipo mede KLD/top-1 contra eles.
Acrescenta linhas em experimentos/resultados/kquant.csv (F16 incluido como referencia).

ATENCAO: a PPL do llama-perplexity NAO e comparavel em valor absoluto com a do MLX
(avaliar.py): o llama-perplexity divide o texto em janelas de n_ctx=2048 e pontua apenas a
2a metade de cada janela (contexto de 1024+ tokens), enquanto avaliar.py pontua todos os
tokens de janelas disjuntas; alem disso o texto e tokenizado pelo proprio llama.cpp.
Compare apenas razoes contra o proprio F16 (ppl_tipo / ppl_F16, kld_media, top1_igual).
A PPL do F16 no CSV e a "Mean PPL(base)" do modo --kl-divergence (mesma contagem de
tokens das demais linhas); o "Final estimate" do passe que grava os logits-base difere
ligeiramente e e ignorado.
Os logits-base ocupam ~45 GB em disco e sao apagados ao final (--manter-logits evita).

Ferramentas: o conversor roda com o .venv do projeto usando PYTHONPATH=<llama.cpp>/gguf-py
(nada e instalado no .venv). Compilar (uma vez):
    export PATH=$HOME/.local/bin:$PATH CC="$(xcrun -f clang)" CXX="$(xcrun -f clang++)"
    cd experimentos/ferramentas/llama.cpp
    cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build build -j 12 --target llama-quantize llama-imatrix llama-perplexity llama-bench

VELOCIDADE (NAO executada aqui: a GPU e compartilhada com outras sessoes; rodar com a
maquina ociosa; -r 5 repeticoes, pp512/tg128 como no MLX):
    experimentos/ferramentas/llama.cpp/build/bin/llama-bench \
        -m experimentos/ferramentas/gguf/<modelo>/<modelo>-<TIPO>.gguf \
        -p 512 -n 128 -r 5 -ngl 99 -o csv
"""
import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
LLAMA = RAIZ / "experimentos/ferramentas/llama.cpp"
BIN = LLAMA / "build/bin"
CALIB = Path.home() / ".cache/mlx-lm/calibration_v5.txt"
CSV = RAIZ / "experimentos/resultados/kquant.csv"
NOMINAL = {"F16": 16, "Q8_0": 8, "Q4_K_M": 4, "Q3_K_M": 3, "Q2_K": 2}
CAMPOS = ["modelo", "metodo", "tipo", "bits_nominais", "bits_por_peso", "tamanho_gb",
          "ppl", "ppl_erro", "kld_media", "top1_igual", "llamacpp_versao", "data"]


def run(cmd, log):
    print("+", " ".join(map(str, cmd)), flush=True)
    with open(log, "w") as f:
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    txt = Path(log).read_text()
    if r.returncode:
        sys.exit(f"falhou ({r.returncode}); veja {log}\n{txt[-2000:]}")
    return txt


def texto_wikitext(destino):
    if not destino.exists():
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        destino.write_text("\n\n".join(ds["text"]), encoding="utf-8")
    return destino


def n_parametros(gguf):
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    from gguf import GGUFReader
    return sum(int(t.n_elements) for t in GGUFReader(str(gguf)).tensors)


def versao():
    tag = subprocess.run(["git", "-C", LLAMA, "describe", "--tags", "--always"],
                         capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "-C", LLAMA, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return f"{tag} ({sha})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--tipos", default="Q8_0,Q4_K_M,Q3_K_M,Q2_K")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--manter-logits", action="store_true")
    ap.add_argument("--refazer", action="store_true", help="ignora artefatos existentes")
    a = ap.parse_args()
    tipos = a.tipos.split(",")

    from huggingface_hub import snapshot_download
    snap = snapshot_download(a.modelo)
    nome = a.modelo.split("/")[-1]
    d = RAIZ / "experimentos/ferramentas/gguf" / nome
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(LLAMA / "gguf-py"))

    f16 = d / f"{nome}-F16.gguf"
    if a.refazer or not f16.exists():
        print("+ convert_hf_to_gguf", flush=True)
        r = subprocess.run([sys.executable, LLAMA / "convert_hf_to_gguf.py", snap,
                            "--outtype", "f16", "--outfile", f16], env=env)
        if r.returncode:
            sys.exit("conversao falhou")

    imat = d / "imatrix.gguf"
    if a.refazer or not imat.exists():
        run([BIN / "llama-imatrix", "-m", f16, "-f", CALIB, "-o", imat, "-ngl", 99],
            d / "imatrix.log")

    wiki = texto_wikitext(d.parent / "wikitext2-raw-test.txt")
    base = d / "logits-F16.bin"
    ppl_args = ["-f", wiki, "-c", a.ctx, "-ngl", 99]

    linhas = []
    log = run([BIN / "llama-perplexity", "-m", f16, *ppl_args, "--kl-divergence-base", base],
              d / "ppl-F16.log")
    ref = None

    for t in tipos:
        q = d / f"{nome}-{t}.gguf"
        if a.refazer or not q.exists():
            run([BIN / "llama-quantize", "--imatrix", imat, f16, q, t], d / f"quant-{t}.log")
        log = run([BIN / "llama-perplexity", "-m", q, *ppl_args, "--kl-divergence",
                   "--kl-divergence-base", base], d / f"ppl-{t}.log")
        ppl = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*(?:\+/-|±)\s*([\d.]+)", log)
        if ref is None:
            m = re.search(r"Mean PPL\(base\)\s*:\s*([\d.]+)\s*(?:\+/-|±)\s*([\d.]+)", log)
            ref = ("F16", f16, float(m[1]), float(m[2]), 0.0, 100.0)
            linhas.insert(0, ref)
        kld = re.search(r"Mean\s+KLD:\s*([\d.]+)", log)
        top1 = re.search(r"Same top p:\s*([\d.]+)", log)
        linhas.append((t, q, float(ppl[1]), float(ppl[2]), float(kld[1]), float(top1[1])))

    if not a.manter_logits:
        base.unlink(missing_ok=True)
    ver, hoje = versao(), datetime.date.today().isoformat()
    novo = not CSV.exists()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(CAMPOS)
        for t, arq, ppl, err, kld, top1 in linhas:
            n = n_parametros(arq)
            tam = arq.stat().st_size
            w.writerow([a.modelo, "kquant", t, NOMINAL.get(t, ""), f"{tam * 8 / n:.3f}",
                        f"{tam / 1e9:.4f}", f"{ppl:.4f}", f"{err:.4f}", f"{kld:.6f}",
                        f"{top1:.3f}", ver, hoje])
            print(f"{t:7s} ppl={ppl:.4f}±{err:.4f} kld={kld:.5f} top1={top1:.2f}% "
                  f"{tam * 8 / n:.2f} b/peso {tam / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
