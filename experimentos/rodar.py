"""Quantiza (RTN/GPTQ/AWQ/DWQ) ou converte (fp) um modelo e grava metricas em resultados/resultados.csv.

Uso:
  python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo awq --bits 4 --seed 0 --fase tudo
Fases: quantizar | qualidade (quantiza se faltar) | velocidade (exige o modelo ja quantizado) | tudo.
O modelo quantizado fica em experimentos/modelos/<id> (ignorado pelo git) para a fase de velocidade
rodar depois, isolada. Uma linha do CSV por (modelo, metodo, bits, grupo, seed); as fases preenchem
colunas diferentes da mesma linha. Gravacao protegida por lock (seguro com varios processos).
"""
import argparse
import csv
import datetime
import fcntl
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
MODELOS = RAIZ / "modelos"
CSV = RAIZ / "resultados" / "resultados.csv"
NLL = RAIZ / "resultados" / "nll"
PY = sys.executable
CHAVE = ["modelo", "metodo", "bits", "grupo", "seed"]
COLUNAS = ["data", "commit", *CHAVE, "tamanho_gb", "tempo_quant_s", "ppl_wikitext2", "janelas", "tokens_avaliados", "seq",
           "acc_arc_easy", "acc_arc_easy_ep", "acc_arc_challenge", "acc_arc_challenge_ep", "acc_hellaswag",
           "acc_hellaswag_ep", "acc_piqa", "acc_piqa_ep", "acc_winogrande", "acc_winogrande_ep", "acc_media", "pico_memoria_ppl_gb",
           "prefill_tps", "prefill_tps_dp", "decode_tps", "decode_tps_dp", "pico_memoria_gb",
           "status", "mlx_lm", "mlx", "python"]


def identificador(modelo, metodo, bits, grupo, seed):
    base = modelo.split("/")[-1] + "-" + metodo
    if metodo == "fp":
        return base
    return f"{base}-{bits}b-g{grupo}-s{seed}"


def commit():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True)
    suja = subprocess.run(["git", "status", "--porcelain", "--", "experimentos/*.py"], cwd=RAIZ.parent,
                          capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+sujo" if suja else "")


def gravar(linha, csv_path=CSV):
    """Insere ou atualiza a linha com a mesma chave, sob lock exclusivo."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        linhas = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
        chave = lambda d: tuple(str(d.get(k, "")) for k in CHAVE)
        for l in linhas:
            if chave(l) == chave(linha):
                l.update({k: v for k, v in linha.items() if v != ""})
                break
        else:
            linhas.append(linha)
        extras = [k for l in linhas for k in l if k not in COLUNAS]
        campos = COLUNAS + sorted(set(extras))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos, restval="")
            w.writeheader()
            w.writerows(linhas)


def existe(linha_chave, coluna, csv_path=CSV):
    if not csv_path.exists():
        return False
    ch = tuple(str(linha_chave[k]) for k in CHAVE)
    return any(tuple(r.get(k, "") for k in CHAVE) == ch and r.get(coluna, "") != "" for r in csv.DictReader(open(csv_path)))


def gptq_camada():
    return os.environ.get("EAQ_GPTQ_CAMADA") == "1"


def awq_liberar():
    return os.environ.get("EAQ_AWQ_LIBERAR") == "1"


def quantizar(modelo, metodo, bits, grupo, seed, destino):
    base = [PY, "-m", "mlx_lm"]
    d = str(destino)
    if metodo == "fp":
        cmd = base + ["convert", "--hf-path", modelo, "--mlx-path", d, "--dtype", "bfloat16"]
    elif metodo == "rtn":
        cmd = base + ["convert", "--hf-path", modelo, "--mlx-path", d, "-q", "--q-bits", str(bits),
                      "--q-group-size", str(grupo), "--dtype", "bfloat16"]
    elif metodo == "gptq" and gptq_camada():
        # mesma quantizacao do mlx-lm (704/704 tensores identicos em Qwen3-0,6B e 1,7B), Hessiana bf16 como no padrao,
        # mas uma camada por vez (pico 3-4x menor). Ativado por EAQ_GPTQ_CAMADA=1; ver protocolo-v2.
        cmd = [PY, str(RAIZ / "gptq_camada.py"), "--model", modelo, "--mlx-path", d, "--bits", str(bits),
               "--group-size", str(grupo), "--seed", str(seed), "--hessiana", "bf16"]
    elif metodo == "awq" and awq_liberar():
        # mesmo AWQ do mlx-lm (704/704 tensores identicos e NLL identica em Qwen3-0,6B e 1,7B), mas liberando a memoria ao fim de
        # cada bloco (pico 3x menor). Ativado por EAQ_AWQ_LIBERAR=1 ou --awq-liberar; ver protocolo-v2.
        cmd = [PY, str(RAIZ / "awq_liberar.py"), "--model", modelo, "--mlx-path", d, "--bits", str(bits),
               "--group-size", str(grupo), "--seed", str(seed)]
    elif metodo in ("gptq", "awq", "dwq"):
        cmd = base + [metodo, "--model", modelo, "--mlx-path", d, "--bits", str(bits),
                      "--group-size", str(grupo), "--seed", str(seed)]
        if metodo == "dwq":
            cmd.append("--grad-checkpoint")  # so reduz memoria (recalcula ativacoes); nao muda o resultado
    else:
        raise SystemExit(f"metodo desconhecido: {metodo}")
    t0 = datetime.datetime.now()
    # /usr/bin/time -l mede o pico real de memoria da quantizacao (peak memory footprint, inclui a GPU/Metal); a saida de
    # erro do job (barras de progresso) continua indo ao log em tempo real. Serve para validar matriz.estimativa_gb.
    p = subprocess.Popen(["/usr/bin/time", "-l", *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    fim = b""
    while chunk := p.stderr.read1(65536):
        sys.stderr.buffer.write(chunk)
        sys.stderr.buffer.flush()
        fim = (fim + chunk)[-8192:]
    rc = p.wait()
    m = re.search(rb"(\d+)\s+peak memory footprint", fim)
    if m:
        print(f"[pico_quant] {metodo} {modelo} {bits}b: footprint {int(m.group(1)) / 1e9:.2f} GB", file=sys.stderr, flush=True)
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    return (datetime.datetime.now() - t0).total_seconds()


def tamanho_gb(pasta):
    return sum(f.stat().st_size for f in Path(pasta).rglob("*") if f.is_file()) / 1e9


def avaliar(destino, fase, nll=None, limite=None):
    cmd = [PY, str(RAIZ / "avaliar.py"), str(destino), "--fase", fase]
    if nll:
        cmd += ["--nll-saida", str(nll)]
    if limite:
        cmd += ["--limite", str(limite)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit(f"avaliar.py falhou ({fase})")
    return json.loads(r.stdout.strip().splitlines()[-1])


def arredondar(d):
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--metodo", required=True, choices=["fp", "rtn", "gptq", "awq", "dwq"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--grupo", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fase", default="tudo", choices=["quantizar", "qualidade", "velocidade", "tudo"])
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--limite", type=int, help="limita exemplos por tarefa (so testes)")
    ap.add_argument("--refazer", action="store_true", help="ignora resultados ja gravados")
    ap.add_argument("--apagar", action="store_true", help="apaga o modelo quantizado no fim")
    ap.add_argument("--gptq-camada", action="store_true",
                    help="GPTQ pelo gptq_camada.py (identico ao mlx-lm, bem menos memoria); equivale a EAQ_GPTQ_CAMADA=1")
    ap.add_argument("--awq-liberar", action="store_true",
                    help="AWQ pelo awq_liberar.py (identico ao mlx-lm, memoria liberada por bloco); equivale a EAQ_AWQ_LIBERAR=1")
    a = ap.parse_args()
    if a.gptq_camada:
        os.environ["EAQ_GPTQ_CAMADA"] = "1"
    if a.awq_liberar:
        os.environ["EAQ_AWQ_LIBERAR"] = "1"

    bits = 16 if a.metodo == "fp" else a.bits
    grupo = "" if a.metodo == "fp" else a.grupo
    seed = a.seed if a.metodo in ("gptq", "awq", "dwq") else 0  # RTN e FP sao deterministicos
    chave = dict(modelo=a.modelo, metodo=a.metodo, bits=bits, grupo=grupo, seed=seed)
    ident = identificador(a.modelo, a.metodo, bits, grupo, seed)
    destino = MODELOS / ident
    base = dict(data=datetime.date.today().isoformat(), commit=commit(), **chave,
                mlx_lm=md.version("mlx-lm"), mlx=md.version("mlx"), python=platform.python_version())
    if a.metodo == "gptq" and gptq_camada():
        base["impl"] = "gptq_camada.py (Hessiana bf16, identico ao mlx-lm)"   # coluna extra: rastreia a implementacao usada
    if a.metodo == "awq" and awq_liberar():
        base["impl"] = "awq_liberar.py (gc por bloco, identico ao mlx-lm)"

    if a.fase in ("quantizar", "qualidade", "tudo") and not (destino / "config.json").exists():
        MODELOS.mkdir(exist_ok=True)
        if destino.exists():
            shutil.rmtree(destino)
        t = quantizar(a.modelo, a.metodo, bits, grupo, seed, destino)
        gravar({**base, "tempo_quant_s": round(t, 1), "tamanho_gb": round(tamanho_gb(destino), 3),
                "status": "quantizado"}, a.csv)
    elif a.fase in ("quantizar", "qualidade", "tudo") and not existe(chave, "tamanho_gb", a.csv):
        gravar({**base, "tamanho_gb": round(tamanho_gb(destino), 3), "status": "quantizado"}, a.csv)
    if a.fase in ("qualidade", "tudo") and (a.refazer or not existe(chave, "acc_media", a.csv)):
        NLL.mkdir(parents=True, exist_ok=True)
        m = avaliar(destino, "qualidade", NLL / f"{ident}.npy", a.limite)
        gravar({**base, **arredondar(m), "status": "qualidade"}, a.csv)
    if a.fase in ("velocidade", "tudo") and (a.refazer or not existe(chave, "decode_tps", a.csv)):
        m = avaliar(destino, "velocidade")
        gravar({**base, **arredondar(m), "status": "completo"}, a.csv)
    if a.apagar:
        shutil.rmtree(destino, ignore_errors=True)
    print(f"ok {ident}")


if __name__ == "__main__":
    main()
