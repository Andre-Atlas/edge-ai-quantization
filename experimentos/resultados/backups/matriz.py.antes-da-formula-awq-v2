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
        lista.append(dict(modelo=m["id"], metodo="fp", bits=16, grupo="", seed=0, fp_gb=m["fp_gb"], arq=m.get("arq")))
        for met, bits in itertools.product(cfg["metodos"], cfg["bits"]):
            if metodos and met not in metodos:
                continue
            for seed in cfg["metodos"][met]["seeds"]:
                lista.append(dict(modelo=m["id"], metodo=met, bits=bits, grupo=cfg["grupo"], seed=seed, fp_gb=m["fp_gb"], arq=m.get("arq")))
    r = cfg.get("replicas")
    if r:
        for m in cfg["modelos"]:
            if m["id"] not in r["modelos"] or (somente and m["id"] not in somente):
                continue
            for met, bits, seed in itertools.product(r["metodos"], r["bits"], r["seeds"]):
                if not metodos or met in metodos:
                    lista.append(dict(modelo=m["id"], metodo=met, bits=bits, grupo=cfg["grupo"], seed=seed, fp_gb=m["fp_gb"], arq=m.get("arq")))
    return [j for j in lista if not excluido(cfg, j)]


def excluido(cfg, j):
    """Combinacoes que a ferramenta nao suporta (matriz.json, chave 'excluir'); campos ausentes casam com qualquer valor."""
    return any(all(j[k] == e[k] for k in ("modelo", "metodo", "bits") if k in e) for e in cfg.get("excluir", []))


def estimativa_gb(j):
    """Pico estimado de memoria de GPU do job (GB).

    Quantizacao com calibracao carrega o modelo bf16 (fp_gb) e ativacoes; o DWQ carrega professor
    e aluno e treina. A avaliacao de PPL soma ~4 GB de logits (vocabulario ~150k, uma janela).
    Estimativa conservadora de proposito: subestimar travou o Mac (22/10/2026, ver README).
    fp e rtn: pico_fp_rtn_gb (calibrado, 25/10/2026). GPTQ e AWQ: pico_gptq_awq_gb (calibrado, em footprint do processo).
    DWQ NAO foi calibrado: as duas medicoes (Qwen3-0,6B e 1,7B) foram abortadas pela trava de seguranca e so deram
    limites inferiores (pico do MLX > 6,1 GB e > 9,6 GB, ambos abaixo desta formula antiga); mantem-se fp_gb * 3 + 4.
    """
    if j["metodo"] in ("fp", "rtn"):
        return pico_fp_rtn_gb(j) + 1.5
    if usa_camada(j):
        return pico_gptq_camada_gb(j)
    if j["metodo"] in ("gptq", "awq"):
        return pico_gptq_awq_gb(j)
    if j["metodo"] == "dwq":
        return j["fp_gb"] * 3 + 4
    return j["fp_gb"] * 1.5 + 4


def usa_camada(j):
    """GPTQ cujo pico na implementacao padrao passa da RAM roda com gptq_camada.py (identico ao mlx-lm, 3-4x menos memoria)."""
    return j["metodo"] == "gptq" and pico_gptq_awq_gb(j) > memoria_total_gb()


def pico_gptq_camada_gb(j):
    """Footprint do gptq_camada.py: 2,54 GB no Qwen3-0,6B e 6,87 GB no 1,7B (2,1 e 1,7 x fp_gb); usa-se 2,1 x fp_gb + 0,5.
    Extrapolacao para 3B e 4B (SmolLM3, Llama, Qwen3-4B): nao validada, a confrontar com o [pico_quant] do primeiro job."""
    return 2.1 * j["fp_gb"] + 0.5


def hessianas_gb(arq):
    """Memoria das Hessianas do GPTQ: um acumulador float32 (in_features^2) por nn.Linear de cada camada
    (q, k, v, gate, up com entrada d; o_proj com entrada o_in; down_proj com entrada i)."""
    return 4 * arq["n"] * (5 * arq["d"] ** 2 + arq["o_in"] ** 2 + arq["i"] ** 2) / 1e9


def pico_gptq_awq_gb(j):
    """Pico de memoria (footprint do processo, GB) de GPTQ ou AWQ, calibrado em 25/10/2026 (pico-quantizacao.csv,
    pico-footprint.csv; 4 bits, grupo 64, Qwen3-0,6B e 1,7B). Faixa de validade: so ha medicao nesses dois modelos;
    fora dela e extrapolacao, a validar com as linhas [pico_quant] dos logs (GPTQ do Llama, replicas de AWQ).

    GPTQ: pesos + 1,5 x Hessianas + 1,5 GB (medido no MLX: 5,8 GB no 0,6B e 14,7 GB no 1,7B; a conta estrutural, sem a
          constante, subestimava o 0,6B em 1,4 GB). As Hessianas dominam (acumuladores float32 + produtos bf16 do lote).
    AWQ:  9,0 + 3,75 x fp_gb (MLX: 13,4 GB no 0,6B e 24,2 GB no 1,7B; quase nao depende do tamanho do modelo).
    Do pico do MLX ao footprint: x1,4 (o footprint do GPTQ no 0,6B foi 8,1 GB contra 5,8 do MLX; nos jobs leves a
    diferenca foi de 0,6 a 1,3 GB).
    """
    if j["metodo"] == "gptq":
        h = hessianas_gb(j["arq"]) if j.get("arq") else 3.5 * j["fp_gb"]
        mlx = j["fp_gb"] + 1.5 * h + 1.5
    else:
        # AWQ rodado SEMPRE pelo awq_liberar.py (identico ao mlx-lm; a matriz passa --awq-liberar): footprint medido de 5,43 GB
        # no Qwen3-0,6B e 9,30 GB no 1,7B (padrao: 15,6 e ~24+ GB). 4,0 + 1,4 x fp_gb cobre os dois pontos (5,7 e 9,7 GB).
        # Extrapolacao a 3B e 4B (Llama ~13 GB, Qwen3-4B ~15 GB): nao validada; confrontar com o [pico_quant] do primeiro job.
        return 4.0 + 1.4 * j["fp_gb"]
    return mlx * 1.4


def pico_fp_rtn_gb(j):
    """Pico de memoria de GPU de um job fp ou rtn (GB), calibrado com medicoes (recalibracao de 25/10/2026).

    Avaliacao de PPL: pico = tamanho do modelo na largura de bits + 1,9 a 2,7 GB (22 linhas de resultados.csv,
    Qwen3-0,6B/1,7B/4B, SmolLM3, Llama; tamanho do rtn = fp_gb * (bits + 0,5) / 16, com escala e bias por grupo 64).
    Conversao (mlx_lm convert): pico <= tamanho bf16 (medido em 0,6B e 1,7B, pico-quantizacao.csv); usa-se fp_gb + 0,5.
    O que for maior dos dois; antes usava-se fp_gb * 1,5 + 4, que exigia 14,1 GB livres no Qwen3-4B e travou a fila
    por ~4,5 h com 12,6 GB livres (picos medidos de fp: 10,5 GB; de rtn: 4,4 a 6,7 GB).
    """
    tamanho = j["fp_gb"] if j["metodo"] == "fp" else j["fp_gb"] * (j["bits"] + 0.5) / 16
    return max(j["fp_gb"] + 0.5, tamanho + 2.7)


def necessario_sozinho_gb(j):
    """Memoria livre minima para iniciar o job mesmo sem outros rodando: pesos + logits + 2 GB de folga."""
    if j["metodo"] in ("fp", "rtn"):
        return pico_fp_rtn_gb(j) + 2
    if j["metodo"] in ("gptq", "awq"):
        # jobs maiores que a RAM rodam sozinhos com swap (SmolLM3 GPTQ, AWQ 1,7B: sem travar); exigir o pico inteiro livre os
        # bloquearia para sempre. Teto: metade da RAM livre, o que ja funcionou como limiar de partida nos jobs ate hoje.
        return min(estimativa_gb(j) + 2, 0.5 * memoria_total_gb())
    return (j["fp_gb"] * 2 if j["metodo"] == "dwq" else j["fp_gb"]) + 4 + 2


def memoria_total_gb():
    import psutil

    return psutil.virtual_memory().total / 1e9


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
    if usa_camada(j):
        c.append("--gptq-camada")
    if j["metodo"] == "awq":
        c.append("--awq-liberar")
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
