"""Testes de validacao do pipeline (Etapa 2). Rodam no Qwen3-0.6B, ~10-15 min.

  python experimentos/testes.py
Usa um CSV temporario; nao altera resultados/resultados.csv.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
MODELO = "Qwen/Qwen3-0.6B"
TMP = Path(tempfile.mkdtemp()) / "t.csv"
falhas = []


def rodar(metodo, bits=4, fase="qualidade", extra=()):
    cmd = [sys.executable, str(RAIZ / "rodar.py"), "--modelo", MODELO, "--metodo", metodo, "--bits", str(bits),
           "--fase", fase, "--csv", str(TMP), "--refazer", "--limite", "40", *extra]
    subprocess.run(cmd, check=True, capture_output=True)
    for l in csv.DictReader(open(TMP)):
        if l["metodo"] == metodo and l["bits"] == str(16 if metodo == "fp" else bits):
            return l
    raise RuntimeError("linha nao encontrada")


def checar(nome, ok, detalhe):
    print(("PASSOU " if ok else "FALHOU ") + nome + ": " + detalhe, flush=True)
    if not ok:
        falhas.append(nome)


fp = rodar("fp")
ppl = {b: float(rodar("rtn", b)["ppl_wikitext2"]) for b in (8, 4, 3, 2)}
ppl_fp = float(fp["ppl_wikitext2"])

r2 = float(rodar("rtn", 4)["ppl_wikitext2"])
checar("determinismo", abs(r2 - ppl[4]) < 1e-3, f"RTN 4b: {ppl[4]:.4f} vs {r2:.4f}")
checar("monotonicidade", ppl[8] <= ppl[4] <= ppl[3] <= ppl[2], " <= ".join(f"{ppl[b]:.2f}" for b in (8, 4, 3, 2)))
checar("8 bits ~ FP (<=1%)", ppl[8] <= ppl_fp * 1.01, f"{ppl[8]:.3f} vs fp {ppl_fp:.3f}")

tam_fp = float(fp["tamanho_gb"])
for b in (8, 4, 3, 2):
    teor = tam_fp / 2 * (b + 32 / 64) / 8
    real = float(rodar("rtn", b, "quantizar")["tamanho_gb"])
    checar(f"tamanho {b}b (+-10% do teorico)", abs(real - teor) / teor < 0.10, f"{real:.3f} GB vs {teor:.3f} GB")

acc = [float(fp[f"acc_{t}"]) for t in ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande")]
checar("lm-eval (fumaca)", all(0 < a <= 1 for a in acc) and sum(acc) / 5 > 0.3, "acc=" + ", ".join(f"{a:.2f}" for a in acc))

print("\nTODOS OS TESTES PASSARAM" if not falhas else f"\nFALHARAM: {falhas}")
sys.exit(1 if falhas else 0)
