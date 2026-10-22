"""Executa a matriz de experimentos (matriz.json) de forma retomavel.

  python experimentos/matriz.py --fase qualidade --workers 3 [--somente Qwen/Qwen3-0.6B] [--metodos rtn,awq] [--simular]
  python experimentos/matriz.py --fase velocidade          # sempre 1 job por vez, maquina ociosa

Fase qualidade: quantiza + PPL + lm-eval, em paralelo, com guarda de memoria (soma das estimativas
<= memoria_total_gb). Fase velocidade: tok/s e pico de memoria, em serie e SEM outros jobs, porque
processos concorrentes dividem GPU e banda de memoria e contaminariam a medida (H4).
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rodar  # noqa: E402

RAIZ = Path(__file__).resolve().parent
LOGS = RAIZ / "logs"


def jobs(cfg, somente=None, metodos=None):
    lista = []
    for m in cfg["modelos"]:
        if somente and m["id"] not in somente:
            continue
        lista.append(dict(modelo=m["id"], metodo="fp", bits=16, grupo="", seed=0, fp_gb=m["fp_gb"]))
        for met, bits in itertools.product(cfg["metodos"], cfg["bits"]):
            if metodos and met not in metodos:
                continue
            for seed in cfg["metodos"][met]["seeds"]:
                lista.append(dict(modelo=m["id"], metodo=met, bits=bits, grupo=cfg["grupo"], seed=seed, fp_gb=m["fp_gb"]))
    r = cfg.get("replicas")
    if r:
        for m in cfg["modelos"]:
            if m["id"] not in r["modelos"] or (somente and m["id"] not in somente):
                continue
            for met, bits, seed in itertools.product(r["metodos"], r["bits"], r["seeds"]):
                if not metodos or met in metodos:
                    lista.append(dict(modelo=m["id"], metodo=met, bits=bits, grupo=cfg["grupo"], seed=seed, fp_gb=m["fp_gb"]))
    return lista


def estimativa_gb(j):
    """Pico estimado de memoria de GPU do job (GB).

    Quantizacao com calibracao carrega o modelo bf16 (fp_gb) e ativacoes; o DWQ carrega professor
    e aluno e treina. A avaliacao de PPL soma ~4 GB de logits (vocabulario ~150k, uma janela).
    Estimativa conservadora de proposito: subestimar travou o Mac (22/10/2026, ver README).
    """
    if j["metodo"] == "dwq":
        return j["fp_gb"] * 3 + 4
    return j["fp_gb"] * 1.5 + 4


def necessario_sozinho_gb(j):
    """Memoria livre minima para iniciar o job mesmo sem outros rodando: pesos + logits + 2 GB de folga."""
    return (j["fp_gb"] * 2 if j["metodo"] == "dwq" else j["fp_gb"]) + 4 + 2


def memoria_livre_gb():
    import psutil

    return psutil.virtual_memory().available / 1e9


def pendente(j, fase):
    col = "acc_media" if fase == "qualidade" else "decode_tps"
    return not rodar.existe({k: j[k] for k in rodar.CHAVE}, col)


def comando(j, fase):
    c = [sys.executable, str(RAIZ / "rodar.py"), "--modelo", j["modelo"], "--metodo", j["metodo"],
         "--seed", str(j["seed"]), "--fase", fase]
    if j["metodo"] != "fp":
        c += ["--bits", str(j["bits"]), "--grupo", str(j["grupo"])]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", required=True, choices=["qualidade", "velocidade"])
    ap.add_argument("--workers", type=int, default=1, help="padrao 1; paralelo so apos medir o ganho (medir_paralelismo.sh)")
    ap.add_argument("--somente", nargs="*")
    ap.add_argument("--metodos", type=lambda s: s.split(","))
    ap.add_argument("--simular", action="store_true", help="apenas lista o que seria executado")
    a = ap.parse_args()
    cfg = json.load(open(RAIZ / "matriz.json"))
    fila = [j for j in jobs(cfg, a.somente, a.metodos) if pendente(j, a.fase)]
    if a.fase == "velocidade":
        a.workers = 1
        fila = [j for j in fila if (rodar.MODELOS / rodar.identificador(j["modelo"], j["metodo"], j["bits"], j["grupo"], j["seed"]) / "config.json").exists()]
    print(f"{len(fila)} jobs pendentes ({a.fase}), workers={a.workers}")
    if a.simular:
        for j in fila:
            print(" ", j["modelo"], j["metodo"], j["bits"], "seed", j["seed"], f"~{estimativa_gb(j):.1f} GB")
        return
    LOGS.mkdir(exist_ok=True)
    rodando, limite, ultimo_aviso = [], cfg["memoria_total_gb"], 0.0
    while fila or rodando:
        for p, j, t0 in rodando[:]:
            if p.poll() is not None:
                rodando.remove((p, j, t0))
                ident = rodar.identificador(j["modelo"], j["metodo"], j["bits"], j["grupo"], j["seed"])
                if p.returncode != 0:
                    rodar.gravar({**{k: j[k] for k in rodar.CHAVE}, "data": time.strftime("%Y-%m-%d"),
                                  "status": f"falhou:{a.fase}"})
                print(f"[{'ok' if p.returncode == 0 else 'FALHOU'}] {ident} ({time.time() - t0:.0f}s)", flush=True)
        usado = sum(estimativa_gb(j) for _, j, _ in rodando)
        if fila and not rodando and memoria_livre_gb() < necessario_sozinho_gb(fila[0]):
            if time.time() - ultimo_aviso > 60:
                print(f"[aguardando memoria] {fila[0]['modelo']} {fila[0]['metodo']} precisa de "
                      f"~{necessario_sozinho_gb(fila[0]):.0f} GB livres, ha {memoria_livre_gb():.1f} GB. "
                      "Feche apps pesados (Teams, Outlook, Chrome...).", flush=True)
                ultimo_aviso = time.time()
            time.sleep(10)
            continue
        # alem do orcamento, so inicia outro job em paralelo se a memoria livre real comportar + 6 GB
        # de folga para o macOS (WindowServer tambem usa a GPU)
        while fila and len(rodando) < a.workers and (not rodando or (usado + estimativa_gb(fila[0]) <= limite
                                                                      and memoria_livre_gb() >= estimativa_gb(fila[0]) + 6)):
            j = fila.pop(0)
            ident = rodar.identificador(j["modelo"], j["metodo"], j["bits"], j["grupo"], j["seed"])
            log = open(LOGS / f"{ident}.{a.fase}.log", "w")
            rodando.append((subprocess.Popen(comando(j, a.fase), stdout=log, stderr=subprocess.STDOUT), j, time.time()))
            usado += estimativa_gb(j)
            print(f"[inicio] {ident} (~{estimativa_gb(j):.1f} GB, em uso {usado:.1f})", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
