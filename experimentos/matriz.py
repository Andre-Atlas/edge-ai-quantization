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
    return j["fp_gb"] * (3.0 if j["metodo"] == "dwq" else 2.2) + 1


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
    ap.add_argument("--workers", type=int, default=1)
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
    rodando, limite = [], cfg["memoria_total_gb"]
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
        while fila and len(rodando) < a.workers and (not rodando or usado + estimativa_gb(fila[0]) <= limite):
            j = fila.pop(0)
            ident = rodar.identificador(j["modelo"], j["metodo"], j["bits"], j["grupo"], j["seed"])
            log = open(LOGS / f"{ident}.{a.fase}.log", "w")
            rodando.append((subprocess.Popen(comando(j, a.fase), stdout=log, stderr=subprocess.STDOUT), j, time.time()))
            usado += estimativa_gb(j)
            print(f"[inicio] {ident} (~{estimativa_gb(j):.1f} GB, em uso {usado:.1f})", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
