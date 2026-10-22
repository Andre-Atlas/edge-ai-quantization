(() => {
"use strict";
const NS = "http://www.w3.org/2000/svg";
const METODOS = {
  fp:   {nome:"fp (bf16)", cor:"--m-fp",   forma:"quadrado"},
  rtn:  {nome:"RTN",       cor:"--m-rtn",  forma:"circulo"},
  gptq: {nome:"GPTQ",      cor:"--m-gptq", forma:"triangulo"},
  awq:  {nome:"AWQ",       cor:"--m-awq",  forma:"losango"},
  dwq:  {nome:"DWQ",       cor:"--m-dwq",  forma:"cruz"},
  kquant:{nome:"K-quants", cor:"--m-kq",   forma:"estrela"},
};
const HIP = [
  {id:"H1", t:"Em 4 bits, algum método ciente de dados (GPTQ/AWQ/DWQ) retém ≥ 95% da qualidade do bf16 (Q ≥ 0,95) e o RTN não.",
   r:"em ≥ 3 dos 5 modelos o RTN tem Q ≥ 0,95, ou em ≥ 3 modelos nenhum método ciente de dados atinge Q ≥ 0,95. Corroborada se o critério vale em ≥ 4 dos 5 modelos.", n:"Exige os 5 modelos com RTN, GPTQ, AWQ e DWQ em 4 bits."},
  {id:"H2", t:"A degradação é maior em modelos menores da mesma família: Q(0,6B) < Q(1,7B) < Q(4B).",
   r:"a ordem se confirma (com ICs sem sobreposição) em < 50% das células método × {4, 3 bits}. Corroborada com ≥ 75%.", n:"Exige os três Qwen3 em 4 e 3 bits."},
  {id:"H3", t:"Abaixo de 4 bits a queda é superlinear, e GPTQ ou DWQ a reduzem em relação ao RTN.",
   r:"os incrementos de L = 1 − Q (RTN) não crescem de 4→3→2 bits, ou GPTQ/DWQ não reduzem L em ≥ 20% no 3 bits em ≥ 3 modelos.", n:"Exige RTN em 4/3/2 bits e GPTQ/DWQ em 3 bits nos 5 modelos."},
  {id:"H4", t:"A redução de memória vira ganho de tokens/s quase proporcional: E = (ganho de decode) / (redução de tamanho).",
   r:"E < 0,4 nos modelos de 3–4B em 4 bits. Corroborada se E ≥ 0,6 nos três.", n:"Exige a fase de velocidade, rodada isolada, para SmolLM3-3B, Llama-3.2-3B e Qwen3-4B."},
];
const $ = (s, r=document) => r.querySelector(s);
const el = (tag, attrs={}, kids=[]) => {
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) k==="text" ? e.textContent=v : e.setAttribute(k,v);
  for (const c of [].concat(kids)) e.append(c);
  return e;
};
const se = (tag, attrs={}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
  if (text!=null) e.textContent = text;
  return e;
};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const nf = (v,d=2) => v==null ? "—" : v.toLocaleString("pt-BR",{minimumFractionDigits:d,maximumFractionDigits:d});
const mInfo = m => METODOS[m] || {nome:m, cor:"--m-outro", forma:"circulo"};
const curto = m => (m||"").split("/").pop();
const rotulo = r => r.metodo==="fp" ? "fp" : `${mInfo(r.metodo).nome} ${r.bits}b`;

let DADOS = null;
const filtro = {modelo:new Set(), metodo:new Set(), bits:new Set()};
let ordem = {col:"modelo", dir:1};

/* ---------- tema ---------- */
function iniciarTema() {
  const b = $("#tema"); let t = "auto";
  try { t = localStorage.getItem("tema") || "auto"; } catch {}
  const aplicar = () => {
    if (t==="auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    b.textContent = "Tema: " + ({auto:"auto", light:"claro", dark:"escuro"})[t];
    if (DADOS) desenhar();
  };
  b.addEventListener("click", () => {
    t = {auto:"light", light:"dark", dark:"auto"}[t];
    try { localStorage.setItem("tema", t); } catch {}
    aplicar();
  });
  aplicar();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => DADOS && desenhar());
}

/* ---------- dica (tooltip) ---------- */
const dica = $("#dica");
function mostrarDica(ev, linhas) {
  dica.replaceChildren(...linhas.map(([txt, cls]) => el("div", {class:cls||"", text:txt})));
  dica.hidden = false;
  const r = ev.target.getBoundingClientRect();
  const x = ev.clientX || r.left + r.width/2, y = ev.clientY || r.top;
  const w = dica.offsetWidth, h = dica.offsetHeight;
  dica.style.left = Math.max(8, Math.min(innerWidth - w - 8, x + 12)) + "px";
  dica.style.top = Math.max(8, y - h - 10 < 8 ? y + 16 : y - h - 10) + "px";
}
const esconderDica = () => { dica.hidden = true; };

/* ---------- formas ---------- */
function marca(forma, cx, cy, r, cor) {
  let e;
  const p = pts => se("polygon", {points: pts.map(q=>q.join(",")).join(" ")});
  if (forma==="quadrado") e = se("rect", {x:cx-r*.85, y:cy-r*.85, width:r*1.7, height:r*1.7, rx:2});
  else if (forma==="triangulo") e = p([[cx,cy-r*1.1],[cx+r*1.05,cy+r*.8],[cx-r*1.05,cy+r*.8]]);
  else if (forma==="losango") e = p([[cx,cy-r*1.2],[cx+r*1.2,cy],[cx,cy+r*1.2],[cx-r*1.2,cy]]);
  else if (forma==="cruz") e = p([[cx-r*.4,cy-r],[cx+r*.4,cy-r],[cx+r*.4,cy-r*.4],[cx+r,cy-r*.4],[cx+r,cy+r*.4],[cx+r*.4,cy+r*.4],[cx+r*.4,cy+r],[cx-r*.4,cy+r],[cx-r*.4,cy+r*.4],[cx-r,cy+r*.4],[cx-r,cy-r*.4],[cx-r*.4,cy-r*.4]]);
  else if (forma==="estrela") e = p([[cx,cy-r*1.2],[cx+r*.4,cy-r*.4],[cx+r*1.2,cy],[cx+r*.4,cy+r*.4],[cx,cy+r*1.2],[cx-r*.4,cy+r*.4],[cx-r*1.2,cy],[cx-r*.4,cy-r*.4]]);
  else e = se("circle", {cx, cy, r});
  e.setAttribute("fill", cor); e.setAttribute("class", "pt");
  return e;
}

/* ---------- escalas ---------- */
function nice(min, max, n=5) {
  if (min===max) { min -= 1; max += 1; }
  const raw = (max-min)/n, mag = 10**Math.floor(Math.log10(raw));
  const passo = [1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw);
  const lo = Math.floor(min/passo)*passo, hi = Math.ceil(max/passo)*passo;
  const ticks = []; for (let v=lo; v<=hi+passo/2; v+=passo) ticks.push(+v.toFixed(10));
  return {lo, hi, ticks};
}
const casas = ticks => { const s = ticks.length>1 ? Math.abs(ticks[1]-ticks[0]) : 1; return s>=1 ? 0 : Math.min(4, Math.ceil(-Math.log10(s))+ (s*10**Math.ceil(-Math.log10(s))%1?1:0)); };

function esqueleto(w, h, m) {
  const svg = se("svg", {viewBox:`0 0 ${w} ${h}`, role:"img"});
  return {svg, W:w-m.l-m.r, H:h-m.t-m.b, m};
}
function eixos(svg, g, xs, ys, xl, yl, fx, fy) {
  const {m,W,H} = g;
  ys.ticks.forEach(v => {
    const y = m.t + H - (v-ys.lo)/(ys.hi-ys.lo)*H;
    svg.append(se("line",{class:"grid",x1:m.l,x2:m.l+W,y1:y,y2:y}), se("text",{x:m.l-6,y:y+4,"text-anchor":"end"},fy(v)));
  });
  xs.ticks.forEach(v => {
    const x = m.l + (v-xs.lo)/(xs.hi-xs.lo)*W;
    svg.append(se("text",{x,y:m.t+H+16,"text-anchor":"middle"},fx(v)));
  });
  svg.append(se("line",{class:"ax",x1:m.l,x2:m.l+W,y1:m.t+H,y2:m.t+H}));
  svg.append(se("text",{class:"eixo-t",x:m.l+W/2,y:m.t+H+34,"text-anchor":"middle"},xl));
  svg.append(se("text",{class:"eixo-t",x:12,y:m.t+H/2,"text-anchor":"middle",transform:`rotate(-90 12 ${m.t+H/2})`},yl));
}
const panelW = id => { const hw = $(id).clientWidth || 340, c = Math.max(1, Math.floor((hw+14)/314)); return Math.max(260, Math.floor((hw-14*(c-1))/c)); };
const larg = n => Math.max(280, Math.min(n.clientWidth || 600, 900));

/* ---------- dados filtrados ---------- */
const filtradas = () => DADOS.linhas.filter(r =>
  filtro.modelo.has(r.modelo) && filtro.metodo.has(r.metodo) && filtro.bits.has(String(r.bits)));

function tipPonto(ev, r) {
  mostrarDica(ev, [
    [`${curto(r.modelo)} · ${rotulo(r)}`, "s"],
    [r.razao_ppl!=null ? `razão PPL ${nf(r.razao_ppl,3)}` : "sem razão", ""],
    [`PPL ${nf(r.ppl_wikitext2,2)} · ${nf(r.tamanho_gb,3)} GB`, "s"],
    [`decode ${nf(r.decode_tps,0)} tok/s · pico ${nf(r.pico_memoria_gb,2)} GB`, "s"],
  ]);
}
function ligarDica(e, fn) {
  e.setAttribute("tabindex","0");
  e.addEventListener("pointermove", fn); e.addEventListener("focus", fn);
  e.addEventListener("pointerleave", esconderDica); e.addEventListener("blur", esconderDica);
}

/* ---------- 1. Pareto ---------- */
function pareto(rs) {
  // não dominado: nenhum outro com tamanho <= e razão >=, com pelo menos uma desigualdade estrita
  return rs.filter(a => !rs.some(b => b!==a && b.tamanho_gb<=a.tamanho_gb && b.razao_ppl>=a.razao_ppl &&
    (b.tamanho_gb<a.tamanho_gb || b.razao_ppl>a.razao_ppl)));
}
function graficoPareto() {
  const host = $("#g-pareto"); host.replaceChildren();
  const rs = filtradas().filter(r => r.tamanho_gb!=null && r.razao_ppl!=null);
  if (!rs.length) { host.append(el("p",{class:"vazio",text:"Nenhum ponto com os filtros atuais."})); return; }
  const w = larg(host), h = Math.round(Math.min(380, w*0.7));
  const g = esqueleto(w, h, {l:50,r:16,t:14,b:42});
  const xs = nice(0, Math.max(...rs.map(r=>r.tamanho_gb))*1.08, 5);
  const vals = rs.map(r=>r.razao_ppl);
  const ys = nice(Math.min(...vals,1)-0.04, Math.max(...vals,1)+0.02, 5);
  const X = v => g.m.l + (v-xs.lo)/(xs.hi-xs.lo)*g.W, Y = v => g.m.t + g.H - (v-ys.lo)/(ys.hi-ys.lo)*g.H;
  eixos(g.svg, g, xs, ys, "Tamanho em disco (GB)", "PPL fp / PPL quant.", v=>nf(v,xs.ticks[1]-xs.ticks[0]<1?1:0), v=>nf(v,2));
  const fr = pareto(rs).sort((a,b)=>a.tamanho_gb-b.tamanho_gb);
  if (fr.length>1) g.svg.append(se("polyline",{points:fr.map(r=>`${X(r.tamanho_gb)},${Y(r.razao_ppl)}`).join(" "),fill:"none",stroke:css("--ink2"),"stroke-width":1.5,"stroke-dasharray":"5 4"}));
  rs.forEach(r => {
    const i = mInfo(r.metodo);
    const grp = se("g", {"aria-label":`${curto(r.modelo)} ${rotulo(r)}: razão ${nf(r.razao_ppl,3)}, ${nf(r.tamanho_gb,3)} GB`});
    const cx = X(r.tamanho_gb), cy = Y(r.razao_ppl);
    const alvo = se("circle",{cx,cy,r:14,fill:"transparent",style:"cursor:pointer"});
    grp.append(marca(i.forma,cx,cy,5.5,css(i.cor)), alvo);
    // rótulo direto
    const lab = se("text",{x:cx+9,y:cy-8,style:"font-size:11px"}, (filtro.modelo.size>1 ? curto(r.modelo).replace("Qwen3-","Q3-")+" " : "")+rotulo(r));
    if (cx+9+90 > w) { lab.setAttribute("x",cx-9); lab.setAttribute("text-anchor","end"); }
    grp.append(lab);
    ligarDica(grp, ev => tipPonto(ev, r));
    g.svg.append(grp);
  });
  host.append(g.svg);
  const ps = new Set(fr.map(rotulo));
  host.append(el("p",{class:"rodape",text:`Na fronteira: ${[...new Set(fr.map(r=>curto(r.modelo)+" "+rotulo(r)))].join(", ")}.`}));
}

/* ---------- 2. Qualidade × bits ---------- */
function facetasPorModelo(host, desenhaPainel) {
  host.replaceChildren();
  const modelos = [...filtro.modelo].filter(m => filtradas().some(r=>r.modelo===m)).sort();
  if (!modelos.length) { host.append(el("p",{class:"vazio",text:"Nenhum ponto com os filtros atuais."})); return; }
  modelos.forEach(m => {
    const f = el("div",{class:"faceta"},[el("h4",{text:curto(m)})]);
    desenhaPainel(f, filtradas().filter(r=>r.modelo===m), m);
    host.append(f);
  });
}
function graficoBits() {
  facetasPorModelo($("#g-bits"), (f, rs) => {
    rs = rs.filter(r => r.razao_ppl!=null);
    if (!rs.length) { f.append(el("p",{class:"vazio",text:"Sem razão PPL (falta linha fp deste modelo?)."})); return; }
    const w = panelW('#g-bits'), h = 250;
    const g = esqueleto(w,h,{l:46,r:14,t:12,b:42});
    const bitsU = [...new Set(rs.map(r=>r.bits))].sort((a,b)=>a-b);
    // eixo X categórico, uma posição por bit-width (ordem crescente)
    const px = b => g.m.l + (bitsU.length===1 ? g.W/2 : bitsU.indexOf(b)/(bitsU.length-1)*g.W*0.9+g.W*0.05);
    const vals = rs.map(r=>r.razao_ppl);
    const ys = nice(Math.min(...vals,1)-0.04, Math.max(...vals,1)+0.02, 5);
    const Y = v => g.m.t+g.H-(v-ys.lo)/(ys.hi-ys.lo)*g.H;
    ys.ticks.forEach(v => g.svg.append(se("line",{class:"grid",x1:g.m.l,x2:g.m.l+g.W,y1:Y(v),y2:Y(v)}), se("text",{x:g.m.l-6,y:Y(v)+4,"text-anchor":"end"},nf(v,2))));
    bitsU.forEach(b => g.svg.append(se("text",{x:px(b),y:g.m.t+g.H+16,"text-anchor":"middle"},b+" bits")));
    g.svg.append(se("line",{class:"ax",x1:g.m.l,x2:g.m.l+g.W,y1:g.m.t+g.H,y2:g.m.t+g.H}));
    g.svg.append(se("text",{class:"eixo-t",x:12,y:g.m.t+g.H/2,"text-anchor":"middle",transform:`rotate(-90 12 ${g.m.t+g.H/2})`},"PPL fp / PPL quant."));
    const porMetodo = {};
    rs.forEach(r => (porMetodo[r.metodo] ||= []).push(r));
    // pontos do mesmo bit-width: pequeno deslocamento horizontal por método, para não sobrepor
    const metodos = Object.keys(porMetodo);
    const desloc = m => (metodos.indexOf(m)-(metodos.length-1)/2)*Math.min(14, g.W/(bitsU.length*(metodos.length+1)));
    metodos.forEach(m => {
      const i = mInfo(m), cor = css(i.cor), L = porMetodo[m].sort((a,b)=>a.bits-b.bits);
      if (L.length>1) g.svg.append(se("polyline",{points:L.map(r=>`${px(r.bits)+desloc(m)},${Y(r.razao_ppl)}`).join(" "),fill:"none",stroke:cor,"stroke-width":2}));
      L.forEach(r => {
        const grp = se("g");
        const cx = px(r.bits)+desloc(m), cy = Y(r.razao_ppl);
        grp.append(marca(i.forma,cx,cy,5,cor), se("circle",{cx,cy,r:14,fill:"transparent"}));
        grp.append(se("text",{x:cx,y:cy-10,"text-anchor":"middle"}, nf(r.razao_ppl,3)));
        ligarDica(grp, ev => tipPonto(ev, r));
        g.svg.append(grp);
      });
    });
    f.append(g.svg);
  });
}

/* ---------- 3. Velocidade e memória ---------- */
function barras(f, rs, campo, dpCampo, titulo, unidade, d) {
  const dados = rs.filter(r => r[campo]!=null).sort((a,b)=>(a.metodo==="fp"?-1:b.metodo==="fp"?1:0)||a.bits-b.bits||a.metodo.localeCompare(b.metodo));
  const caixa = el("div",{class:"t",text:titulo});
  f.append(caixa);
  if (!dados.length) { f.append(el("p",{class:"vazio",text:"Sem dados."})); return; }
  const w = panelW('#g-vel'), h = 230;
  const g = esqueleto(w,h,{l:46,r:10,t:12,b:58});
  const topo = Math.max(...dados.map(r => r[campo]+(r[dpCampo]||0)));
  const ys = nice(0, topo*1.05, 4), Y = v => g.m.t+g.H-(v-ys.lo)/(ys.hi-ys.lo)*g.H;
  const dec = ys.ticks.every(t=>Number.isInteger(t)) ? 0 : 1;
  ys.ticks.forEach(v => g.svg.append(se("line",{class:"grid",x1:g.m.l,x2:g.m.l+g.W,y1:Y(v),y2:Y(v)}), se("text",{x:g.m.l-6,y:Y(v)+4,"text-anchor":"end"},nf(v,dec))));
  g.svg.append(se("line",{class:"ax",x1:g.m.l,x2:g.m.l+g.W,y1:Y(0),y2:Y(0)}));
  const slot = g.W/dados.length, bw = Math.min(34, slot*0.62);
  dados.forEach((r,k) => {
    const i = mInfo(r.metodo), cor = css(i.cor), cx = g.m.l+slot*(k+.5), y = Y(r[campo]), h0 = Y(0)-y;
    const grp = se("g");
    const rad = Math.min(4, bw/2, h0);
    grp.append(se("path",{d:`M${cx-bw/2},${Y(0)}V${y+rad}Q${cx-bw/2},${y} ${cx-bw/2+rad},${y}H${cx+bw/2-rad}Q${cx+bw/2},${y} ${cx+bw/2},${y+rad}V${Y(0)}Z`,fill:cor,class:"pt"}));
    const dp = r[dpCampo];
    if (dp) { const a=Y(r[campo]+dp), b=Y(Math.max(0,r[campo]-dp));
      grp.append(se("line",{x1:cx,x2:cx,y1:a,y2:b,stroke:css("--ink"),"stroke-width":1.5}),
        se("line",{x1:cx-4,x2:cx+4,y1:a,y2:a,stroke:css("--ink"),"stroke-width":1.5}),
        se("line",{x1:cx-4,x2:cx+4,y1:b,y2:b,stroke:css("--ink"),"stroke-width":1.5})); }
    grp.append(se("rect",{x:cx-slot/2,y:g.m.t,width:slot,height:g.H,fill:"transparent"}));
    const rot = se("text",{x:cx,y:Y(0)+14,"text-anchor":"end",transform:`rotate(-35 ${cx} ${Y(0)+14})`}, rotulo(r));
    g.svg.append(grp, rot);
    ligarDica(grp, ev => mostrarDica(ev,[[rotulo(r)+" · "+curto(r.modelo),"s"],[`${nf(r[campo],d)} ${unidade}`+(dp?` ± ${nf(dp,d)}`:""),""]]));
  });
  f.append(g.svg);
}
function graficoVel() {
  facetasPorModelo($("#g-vel"), (f, rs, m) => {
    f.className = "faceta"; f.style.gridColumn = "1 / -1";
    const inner = el("div",{class:"facetas"});
    [["decode_tps","decode_tps_dp","Decode (tokens/s)","tok/s",0],
     ["prefill_tps","prefill_tps_dp","Prefill (tokens/s)","tok/s",0],
     ["pico_memoria_gb",null,"Pico de memória (GB)","GB",2]].forEach(([c,dp,t,u,d]) => {
      const sub = el("div",{class:"faceta"}); inner.append(sub);
      sub.style.minWidth = 0;
      inner.append; barras(sub, rs, c, dp, t, u, d);
    });
    f.append(inner);
  });
}

/* ---------- tabela ---------- */
const COLS = [["modelo","Modelo",0],["metodo","Método",0],["bits","Bits",0],["grupo","Grupo",0],["tamanho_gb","GB",3],["ppl_wikitext2","PPL",3],["razao_ppl","Razão",3],["decode_tps","Decode",1],["prefill_tps","Prefill",0],["pico_memoria_gb","Pico GB",2],["tempo_quant_s","Quant. (s)",1]];
function tabela() {
  const t = $("#tabela"); t.replaceChildren();
  const vis = COLS.filter(([c]) => DADOS.colunas.includes(c));
  const th = el("tr");
  vis.forEach(([c,nome]) => {
    const h = el("th",{scope:"col"}); const b = el("button",{type:"button",text:nome});
    if (ordem.col===c) h.setAttribute("aria-sort", ordem.dir>0?"ascending":"descending");
    b.addEventListener("click",()=>{ ordem = {col:c, dir: ordem.col===c ? -ordem.dir : 1}; tabela(); });
    h.append(b); th.append(h);
  });
  t.append(el("thead",{},[th]));
  const rs = filtradas().sort((a,b)=>{
    const x=a[ordem.col], y=b[ordem.col];
    if (x==null) return 1; if (y==null) return -1;
    return (typeof x==="number" ? x-y : String(x).localeCompare(String(y)))*ordem.dir;
  });
  const tb = el("tbody");
  rs.forEach(r => tb.append(el("tr",{},vis.map(([c,,d]) => el("td",{text: typeof r[c]==="number" ? nf(r[c], Number.isInteger(r[c])&&d===0?0:d) : (c==="modelo"?curto(r[c]):(r[c]??"—"))})))));
  t.append(tb);
}

/* ---------- filtros ---------- */
function chips(host, chave, valores, rot, cor) {
  host.replaceChildren();
  valores.forEach(v => {
    const b = el("button",{type:"button",class:"chip","aria-pressed":"true"});
    if (cor) { const i = el("i"); i.style.background = `var(${cor(v)})`; b.append(i); }
    b.append(rot(v));
    b.addEventListener("click", () => {
      const s = filtro[chave]; s.has(v) ? s.delete(v) : s.add(v);
      b.setAttribute("aria-pressed", s.has(v));
      desenhar();
    });
    host.append(b);
  });
}

/* ---------- hipóteses ---------- */
function veredito(id) {
  const h = DADOS.hipoteses; if (!h) return null;
  const item = Array.isArray(h) ? h.find(x => (x.id||x.hipotese)===id)
             : (h.hipoteses ? (Array.isArray(h.hipoteses) ? h.hipoteses.find(x=>(x.id||x.hipotese)===id) : h.hipoteses[id]) : h[id]);
  if (!item) return null;
  if (typeof item === "string") return {v:item};
  return {v:item.veredito ?? item.status ?? item.resultado, resumo:item.resumo ?? item.justificativa ?? item.evidencia};
}
function classe(v) {
  const s = String(v||"").toLowerCase();
  if (/refutad/.test(s)) return ["bad","✕ Refutada"];
  if (/corrobor|sustent|n[aã]o refutad|confirm/.test(s)) return ["ok","✓ "+v];
  if (!s || /aguard|pendente/.test(s)) return ["wait","◔ Aguardando dados"];
  return ["wait","◔ "+v];
}
function hipoteses() {
  const resumo = $("#hip-resumo"), cart = $("#hip-cartoes");
  resumo.replaceChildren(); cart.replaceChildren();
  HIP.forEach(h => {
    resumo.append(el("article",{class:"cartao hcard"},[
      el("h3",{text:h.id}), el("p",{text:h.t}),
      el("p",{},[el("span",{class:"rot",text:"Refutada se "}), document.createTextNode(h.r)])]));
    const v = veredito(h.id), [c, txt] = classe(v && v.v);
    const corpo = [el("h3",{},[document.createTextNode(h.id+" "), el("span",{class:"selo "+c,text:txt})]), el("p",{text:h.t})];
    if (v && v.resumo) corpo.push(el("p",{class:"nota",text:v.resumo}));
    else if (c==="wait") corpo.push(el("p",{class:"nota",text:h.n}));
    cart.append(el("article",{class:"cartao hcard"},corpo));
  });
}

/* ---------- reprodutibilidade ---------- */
function versoes() {
  const host = $("#versoes"); host.replaceChildren();
  const u = c => [...new Set(DADOS.linhas.map(r=>r[c]).filter(Boolean))].join(", ");
  [["mlx-lm","mlx_lm"],["mlx","mlx"],["Python","python"],["Commit","commit"],["Seed","seed"]].forEach(([n,c]) => {
    if (DADOS.colunas.includes(c) && u(c)) host.append(el("span",{text:`${n} ${u(c)}`}));
  });
  host.append(el("span",{text:"Calibração: calibration_v5"}), el("span",{text:"Avaliação: WikiText-2 (teste), seq 2048"}));
  const datas = [...new Set(DADOS.linhas.map(r=>r.data).filter(Boolean))].sort();
  $("#meta-dados").textContent = `${DADOS.n_linhas} linhas, ${datas.join(" – ")}.`;
  if (DADOS.fonte === "matriz") {
    $("#aviso-titulo").textContent = "Matriz do protocolo-v1 (em andamento).";
    $("#aviso-texto").textContent = "Uma semente por configuração (réplicas do AWQ em 4 bits estimam a variância). Resultados parciais enquanto a matriz roda; vereditos só com os dados completos.";
    $("#nota-resultados").textContent = "Q = PPL fp / PPL quantizado (WikiText-2). K-quants: razão contra o F16 do próprio llama.cpp (janelamento diferente do MLX).";
  }
}

function desenhar() { graficoPareto(); graficoBits(); graficoVel(); tabela(); legenda(); }
function legenda() {
  const L = $("#leg-metodos"); L.replaceChildren();
  [...new Set(DADOS.linhas.map(r=>r.metodo))].forEach(m => {
    const i = mInfo(m), s = se("svg",{width:16,height:16,viewBox:"0 0 16 16","aria-hidden":"true"});
    s.append(marca(i.forma,8,8,5,css(i.cor)));
    L.append(el("span",{},[s, document.createTextNode(i.nome)]));
  });
}

async function iniciar() {
  iniciarTema();
  try {
    DADOS = await (await fetch("data/resultados.json")).json();
  } catch (e) {
    $("#g-pareto").append(el("p",{class:"vazio",text:"Não foi possível carregar data/resultados.json."})); return;
  }
  const ms = [...new Set(DADOS.linhas.map(r=>r.metodo))];
  const bs = [...new Set(DADOS.linhas.map(r=>String(r.bits)))].sort((a,b)=>a-b);
  DADOS.modelos.forEach(m=>filtro.modelo.add(m)); ms.forEach(m=>filtro.metodo.add(m)); bs.forEach(b=>filtro.bits.add(b));
  chips($("#f-modelo"),"modelo",DADOS.modelos,curto);
  chips($("#f-metodo"),"metodo",ms,m=>mInfo(m).nome,m=>mInfo(m).cor);
  chips($("#f-bits"),"bits",bs,b=>b+" bits");
  hipoteses(); versoes(); desenhar();
  let t, lw = innerWidth; addEventListener("resize", () => { if (innerWidth===lw) return; lw = innerWidth; clearTimeout(t); t = setTimeout(desenhar, 120); });
}
iniciar();
})();
