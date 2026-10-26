"""Resume a contaminacao da fase de velocidade: CPU do app Claude (amostras de 30 s) e deriva entre os controles.

  python experimentos/resumo_contaminacao.py [AMOSTRAS.log] [etapa4.log]

Usado por rodar_velocidade.sh ao fim da fase. O controle e a MESMA configuracao (Qwen3-0,6B RTN 4 bits) medida 3 vezes
antes e 3 depois; se a deriva for pequena e o CV baixo, a maquina estava estavel durante a fase.
"""
import re
import statistics as st
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
amostras = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "logs/velocidade-amostras.log"
etapa4 = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "logs/etapa4.log"

L = amostras.read_text().splitlines() if amostras.exists() else []
cl = [float(re.search(r"claude_cpu=([\d.]+)", x).group(1)) for x in L if "claude_cpu" in x]
tt = [float(re.search(r"total_cpu=([\d.]+)", x).group(1)) for x in L if "total_cpu" in x]
if cl:
    print(f"[velocidade:amostras] {len(cl)} amostras (30 s): CPU do app Claude media {st.mean(cl):.1f}% (max {max(cl):.1f}%), "
          f"CPU total do sistema media {st.mean(tt):.1f}% (max {max(tt):.1f}%); {sum(1 for v in cl if v > 10)} amostras com o app acima de 10%")
log = [x for x in etapa4.read_text().splitlines() if "[velocidade:controle]" in x]


def decode(rotulo):
    return [float(re.search(r"decode ([\d.]+)", x).group(1)) for x in log if f" {rotulo} #" in x]


a, d = decode("antes"), decode("depois")
cv = lambda v: (100 * st.stdev(v) / st.mean(v)) if len(v) > 1 else 0.0
if a and d:
    ma, md = st.mean(a), st.mean(d)
    print(f"[velocidade:controle] decode antes {ma:.1f} (CV {cv(a):.1f}%) | depois {md:.1f} (CV {cv(d):.1f}%) | deriva {100 * (md - ma) / ma:+.1f}%")
else:
    print("[velocidade:controle] controles ausentes")
