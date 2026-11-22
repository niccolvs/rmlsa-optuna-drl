"""Informe de estudio CORTO del proyecto TEL317. Lee todos los números de results/ (no hay cifras a mano).

Uso (desde la raíz del repo, con el .venv activo y `pip install reportlab`):
    python informe/informe_corto.py [nombre_del_estudio]   (por defecto ppo_rmlsa_100erl_v4 si existe, si no v3)

Sale en informe/Informe_TEL317_Optuna_RMLSA.pdf. Las figuras fijas (NSFNet, espectro, gamma, flujo) están en
informe/fig y se regeneran con informe/figuras.py. Usa las fuentes Arial/Consolas de Windows.
"""
import contextlib
import io
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).parent
FIG = HERE / "fig"
REPO = HERE.parent
RES = REPO / "results"
OUT = HERE / "Informe_TEL317_Optuna_RMLSA.pdf"

# ----------------------------------------------------------------------------- datos
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)
STORAGE = f"sqlite:///{RES / 'optuna.db'}"
names = optuna.get_all_study_names(STORAGE)
STUDY = sys.argv[1] if len(sys.argv) > 1 else ("ppo_rmlsa_100erl_v4" if "ppo_rmlsa_100erl_v4" in names
                                               else "ppo_rmlsa_100erl_v3")
FINAL = RES / "final" / STUDY
OPT = RES / "optuna" / STUDY
study = optuna.load_study(study_name=STUDY, storage=STORAGE)
ok = [t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")]
comp = [t for t in ok if t.state.name == "COMPLETE"]
pruned = [t for t in ok if t.state.name == "PRUNED"]
dur = lambda t: (t.datetime_complete - t.datetime_start).total_seconds()
N, NC, NP = len(ok), len(comp), len(pruned)
total_min = sum(dur(t) for t in study.trials if t.datetime_complete) / 60
saved_min = (np.mean([dur(t) for t in comp]) * NP - sum(dur(t) for t in pruned)) / 60
best = study.best_trial
BP = best.params
imp = optuna.importance.get_param_importances(study)
top = sorted(comp, key=lambda t: t.value)
top10 = top[:10]
after = [t for t in study.trials if t.number >= 10 and t.params]
n_lr_low = sum(t.params["learning_rate"] < 2e-4 for t in after)
n_g_low = sum(t.params["gamma"] < 0.9 for t in after)
EXTENDED = STUDY != "ppo_rmlsa_100erl_v3"

res = pd.read_csv(FINAL / "resultados.csv")
summ = pd.read_csv(FINAL / "resumen.csv")
M = summ.pivot(index="carga_erlang", columns="agente", values="mean")
IC = summ.pivot(index="carga_erlang", columns="agente", values="ic95")
LOADS = list(M.index)
imp_os = pd.read_csv(FINAL / "mejora_PPO_Optuna_vs_SPFF.csv", index_col=0)
imp_od = pd.read_csv(FINAL / "mejora_PPO_Optuna_vs_PPO_defecto.csv", index_col=0)
imp_ds = pd.read_csv(FINAL / "mejora_PPO_defecto_vs_SPFF.csv", index_col=0)
per_train = res[res.agente != "SPFF"].groupby(["agente", "semilla_entrenamiento", "carga_erlang"]).bloqueo.mean()
o100 = per_train["PPO Optuna"].xs(100, level=1)
d100 = per_train["PPO defecto"].xs(100, level=1)
wins_os = int(imp_os.casos_mejor.sum()); wins_od = int(imp_od.casos_mejor.sum()); ncases = int(imp_os["count"].sum())
w = res[res.agente == "PPO Optuna"].groupby(["carga_erlang", "semilla_trafico"]).bloqueo.max()
b = res[res.agente == "PPO defecto"].groupby(["carga_erlang", "semilla_trafico"]).bloqueo.min()
worst_beats_best = int((w < b).sum())
rut = pd.read_csv(FINAL / "red_rutas.csv")
tot = rut.groupby(["agente", "carga_erlang"]).solicitudes.sum()
dev = rut[rut.clave != 0].groupby(["agente", "carga_erlang"]).solicitudes.sum().reindex(tot.index, fill_value=0)
PCT = (dev / tot * 100).unstack(0)


def blocking_by(kind, load=100, bins=None, labels=None):
    d = pd.read_csv(FINAL / f"red_{kind}.csv")
    d = d[d.carga_erlang == load].copy()
    if bins:
        d["clave"] = pd.cut(d.clave, bins=bins, labels=labels)
    g = d.groupby(["agente", "clave"], observed=True)[["solicitudes", "bloqueadas"]].sum()
    return (g.bloqueadas / g.solicitudes).unstack(0)


BR = blocking_by("bitrate")
DI = blocking_by("distancia", bins=[0, 1000, 2000, 3000, 10_000], labels=["< 1000", "1000–2000", "2000–3000", "> 3000"])
enl = pd.read_csv(FINAL / "red_enlaces.csv")
OCC = enl[enl.carga_erlang == 100].groupby("agente").solicitudes.mean()


def first_below(slug, thr=0.10):
    out = []
    for f in sorted((FINAL / "curvas").glob(f"{slug}_s*.csv")):
        c = pd.read_csv(f)
        m = c.bloqueo.rolling(10).mean()
        out.append(int(c.paso[m < thr].min()))
    return out


FB_O, FB_D = first_below("optuna"), first_below("defecto")
eval_min = None
# imagen de la maqueta: un cuadro del video más reciente
video = RES / "maqueta" / ("maqueta_150erl_v4.mp4" if EXTENDED else "maqueta_150erl.mp4")
for alt in [REPO / "evidencias" / "2026-11-20_300trials" / "23_maqueta_150erl.mp4",
            RES / "maqueta" / "maqueta_150erl.mp4",
            REPO / "evidencias" / "2026-11-13_maqueta" / "01_maqueta_150erl.mp4"]:
    if video.exists():
        break
    video = alt
import imageio_ffmpeg  # noqa: E402

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", "30", "-i", str(video), "-frames:v", "1",
                str(FIG / "maqueta_corto.png")])

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x, d=1):
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.{d}f}·10{str(e).translate(SUP)}"


f4 = lambda x: f"{x:.4f}"
pc = lambda x: f"{x:.0f} %"
L = lambda l: f"{l:.0f}"

# ----------------------------------------------------------------------------- estilo
F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Arial", f"{F}\\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-B", f"{F}\\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-I", f"{F}\\ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BI", f"{F}\\arialbi.ttf"))
pdfmetrics.registerFont(TTFont("Mono", f"{F}\\consola.ttf"))
addMapping("Arial", 0, 0, "Arial"); addMapping("Arial", 1, 0, "Arial-B")
addMapping("Arial", 0, 1, "Arial-I"); addMapping("Arial", 1, 1, "Arial-BI")
NAVY, TEAL = colors.HexColor("#1d3557"), colors.HexColor("#2a9d8f")
LIGHT, NOTE, GOOD = colors.HexColor("#eef4f8"), colors.HexColor("#fff6e0"), colors.HexColor("#e6f4ea")
body = ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.6, leading=11.5, alignment=0)
cell = ParagraphStyle("cell", parent=body, fontSize=8.7, leading=11.2, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Arial-B", textColor=colors.white)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Arial-B", fontSize=16.5, leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=9)
h2 = ParagraphStyle("h2", fontName="Arial-B", fontSize=12, leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=4)
code = ParagraphStyle("code", fontName="Mono", fontSize=7.9, leading=10, backColor=colors.HexColor("#f4f5f7"),
                      borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)
caption = ParagraphStyle("cap", parent=small, alignment=TA_CENTER, textColor=colors.HexColor("#555"), spaceAfter=9)
q_style = ParagraphStyle("q", parent=body, fontName="Arial-B", textColor=NAVY, spaceAfter=1, spaceBefore=4)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
FIGN = [0]


def H1(t, newpage=True):
    if newpage:
        story.append(PageBreak())
    else:
        story.append(CondPageBreak(6 * cm))
    p = Paragraph(t, h1); p._toc = (0, t); story.append(p)


def H2(t, need=3):
    story.append(CondPageBreak(need * cm))
    p = Paragraph(t, h2); p._toc = (1, t); story.append(p)


def B(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 3))


def CODE(t):
    story.append(KeepTogether([Preformatted(t.strip("\n"), code)]))


def BOX(t, bg=NOTE):
    tb = Table([[[Paragraph(x, cell) for x in t.split("\n\n")]]], colWidths=[16.4 * cm])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9c9c9")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(KeepTogether([tb, Spacer(1, 8)]))


def TABLE(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    tb = Table(data, colWidths=[w_ * cm for w_ in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5ccd3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    tb.setStyle(TableStyle(st))
    story.append(KeepTogether([tb, Spacer(1, 8)]) if len(rows) <= 14 else tb)
    if len(rows) > 14:
        story.append(Spacer(1, 8))


def FIGURE(path, width_cm, text):
    from reportlab.lib.utils import ImageReader
    FIGN[0] += 1
    iw, ih = ImageReader(str(path)).getSize()
    w_ = width_cm * cm
    story.append(KeepTogether([Image(str(path), width=w_, height=w_ * ih / iw),
                               Paragraph(f"Figura {FIGN[0]}. {text}", caption)]))


def QA(q, a):
    story.append(KeepTogether([Paragraph(q, q_style), Paragraph(a, body)]))


class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if hasattr(f, "_toc"):
            level, text = f._toc
            key = f"s{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level)
            self.notify("TOCEntry", (level, text, self.page, key))


def on_page(c, d):
    if d.page == 1:
        return
    c.saveState()
    c.setFont("Arial", 8); c.setFillColor(colors.HexColor("#777"))
    c.drawString(2.2 * cm, 1.3 * cm, "TEL 317 · Optimización de hiperparámetros con Optuna para RMLSA con DRL")
    c.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, str(d.page))
    c.restoreState()


# ============================================================================ PORTADA + ÍNDICE
story.append(Spacer(1, 4 * cm))
P("TEL 317 · Universidad Técnica Federico Santa María · 2026",
  ParagraphStyle("k", parent=body, alignment=TA_CENTER, textColor=TEAL, fontName="Arial-B"))
P("Optimización de hiperparámetros con Optuna para RMLSA con aprendizaje por refuerzo profundo",
  ParagraphStyle("t", fontName="Arial-B", fontSize=23, leading=29, alignment=TA_CENTER, textColor=NAVY, spaceAfter=16))
P("Guía de estudio del proyecto: qué hicimos, cómo funciona cada pieza, por qué lo decidimos así y qué resultados "
  "obtuvimos", ParagraphStyle("st", parent=body, fontSize=12, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#444")))
story.append(Spacer(1, 1.4 * cm))
P("Andrea Alfonzo · Nicolás Alvarez · Lorenzo Gonzalez", ParagraphStyle("a", parent=body, alignment=TA_CENTER, fontSize=11.5))
P("Documento interno del grupo · versión del 20 de noviembre de 2026",
  ParagraphStyle("d", parent=small, alignment=TA_CENTER, textColor=colors.HexColor("#777")))
story.append(Spacer(1, 2 * cm))
BOX("Este documento es para <b>entender</b> el proyecto y poder explicarlo y defenderlo, no es el informe que se entrega. "
    "Si tienes poco tiempo, lee el resumen (sección 1), los resultados (sección 7) y las preguntas probables (sección 9).\n\n"
    "Código, evidencias y resultados: <b>github.com/niccolvs/rmlsa-optuna-drl</b>.", LIGHT)
story.append(PageBreak())
P("Contenido", h1)
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Arial-B", fontSize=10.5, leading=15),
                   ParagraphStyle("t1", fontName="Arial", fontSize=9.3, leading=12.5, leftIndent=16)]
story.append(toc)

# ============================================================================ 1. RESUMEN
H1("1. Resumen")
P("<b>Qué queríamos.</b> Un agente de aprendizaje por refuerzo profundo (PPO) decide por qué ruta enviar cada solicitud "
  "de conexión que llega a una red óptica elástica. Su desempeño depende de sus <b>hiperparámetros</b>, y elegirlos a mano "
  "es lento y arbitrario. Usamos <b>Optuna</b> para elegirlos automáticamente, minimizando la <b>probabilidad de bloqueo</b> "
  "(fracción de solicitudes rechazadas por falta de espectro), y comparamos el agente optimizado con el mismo agente con "
  "valores por defecto y con la heurística clásica <b>SPFF</b>.")
FIGURE(FIG / "pipeline.png", 14.5, "Flujo del proyecto y objetivo específico que cubre cada etapa.")
P("<b>Qué hicimos.</b>")
B(["Escenario sobre el simulador de <b>DREAM-ON GYM</b>: red NSFNet (14 nodos), 100 slots por enlace, tráfico dinámico, "
   "4 velocidades (10 a 400 Gbps) y 6 modulaciones.",
   "Entorno de aprendizaje propio (<i>RMLSAEnv</i>): el agente ve el estado del espectro en 3 rutas candidatas y elige una.",
   f"Estudio Optuna de <b>{N} trials</b> sobre <i>learning rate</i>, <i>gamma</i> y <i>batch size</i>, cada trial medido "
   "con 3 semillas de tráfico." + (" (Los primeros 100 trials son los del estudio de 100; los otros "
                                   f"{N - 100} lo extienden.)" if EXTENDED else ""),
   f"Evaluación: 3 entrenamientos por configuración, probados en <b>{len(LOADS)} cargas × 10 semillas de tráfico "
   "nuevas</b>, con intervalos de confianza del 95 %.",
   "Análisis de cómo usa la red cada agente, y una <b>maqueta en vivo</b> de NSFNet para la presentación."])
sel = [l for l in [25, 50, 100, 150, 200, 250] if l in LOADS]
TABLE([["Carga [Erlang]", "SPFF", "PPO por defecto", "PPO Optuna", "Mejora vs SPFF", "Mejora vs PPO defecto"]] +
      [[L(l), f4(M.SPFF[l]), f4(M["PPO defecto"][l]), f"<b>{f4(M['PPO Optuna'][l])}</b>", pc(imp_os["mean"][l]),
        pc(imp_od["mean"][l])] for l in sel], [2.6, 2.1, 2.7, 2.4, 2.7, 3.4])
B([f"El agente optimizado bloquea menos que SPFF en {wins_os} de {ncases} casos (carga × semilla) y menos que PPO por "
   f"defecto en {wins_od} de {ncases}, aunque solo entrenó a 100 Erlang.",
   f"Mejores hiperparámetros: <b>gamma = {BP['gamma']:.3f}</b>, <b>learning rate = {sci(BP['learning_rate'])}</b>, "
   f"batch size = {BP['batch_size']} (por defecto: 0.99, 3·10⁻⁴ y 64).",
   "Importancia: " + ", ".join(f"<b>{k.replace('_', ' ')} {v:.2f}</b>" for k, v in imp.items()) +
   (". Gamma es el que más pesa; batch size y learning rate quedan bastante más abajo." if EXTENDED else
    ". Gamma y learning rate son los que importan; el batch size casi no influye."),
   f"<b>Por qué gana:</b> aprendió <i>cuándo</i> desviarse de la ruta más corta: {PCT['PPO Optuna'][LOADS[0]]:.0f} % de "
   f"las veces con la red casi vacía y {PCT['PPO Optuna'][LOADS[-1]]:.0f} % con la red saturada. El agente por defecto "
   f"se desvía ~{PCT['PPO defecto'].mean():.0f} % siempre, sin criterio."])

# ============================================================================ 2. PROBLEMA
H1("2. El problema: redes ópticas elásticas y RMLSA")
P("Una fibra transporta información en un rango de frecuencias (el <b>espectro</b>). Las <b>redes ópticas elásticas</b> "
  "(EON) lo dividen en unidades pequeñas llamadas <b>slots</b> y a cada conexión le asignan solo los que necesita. En "
  "nuestro escenario cada enlace tiene <b>100 slots</b>. Cada vez que llega una solicitud (\"quiero 100 Gbps del nodo 0 al "
  "nodo 3\") la red debe decidir tres cosas:")
TABLE([["Letra", "Decisión", "Qué significa", "En nuestro proyecto"],
       ["R", "Routing", "Por qué camino de enlaces va la conexión", "<b>La decide el agente</b> (1 de 3 rutas)"],
       ["ML", "Modulation Level", "Cuántos bits por símbolo. Más eficiente = menos slots, pero menos alcance",
        "Regla fija: la más eficiente que alcance la distancia"],
       ["SA", "Spectrum Assignment", "Qué slots concretos se usan", "Regla fija: First-Fit (el primer bloque libre)"]],
      [1.3, 3.0, 6.0, 6.1])
P("Dos restricciones físicas lo hacen difícil: <b>continuidad</b> (los mismos slots en todos los enlaces de la ruta) y "
  "<b>contigüidad</b> (los slots deben estar pegados, en un bloque).")
FIGURE(FIG / "espectro.png", 15, "Una conexión de 4 slots en la ruta 0→1→3. Un slot sirve solo si está libre en ambos "
       "enlaces (fila \"unión\"). First-Fit toma el primer bloque de 4 libres seguidos. El hueco del slot 7 queda "
       "inutilizable: eso es <b>fragmentación</b>.")
P("<b>Modulación.</b> Las modulaciones eficientes (64-QAM) usan pocos slots pero llegan cerca; las robustas (BPSK) llegan "
  "lejos pero gastan muchos. Por ejemplo, 100 Gbps en una ruta de 2 090 km usa QPSK (4 slots), pero en una de 4 860 km "
  "solo alcanza BPSK (8 slots). Desviarse a una ruta larga gasta más espectro, pero si la ruta corta está llena puede ser "
  "lo único que evita un bloqueo. <b>Ese dilema es lo que el agente aprende.</b>")
TABLE([["Modulación", "Alcance", "10 Gbps", "40 Gbps", "100 Gbps", "400 Gbps"],
       ["BPSK", "5520 km", "1", "4", "8", "32"], ["QPSK", "2720 km", "1", "2", "4", "16"],
       ["8-QAM", "1360 km", "1", "2", "3", "11"], ["16-QAM", "560 km", "1", "1", "2", "8"],
       ["32-QAM", "240 km", "1", "1", "2", "7"], ["64-QAM", "80 km", "1", "1", "2", "6"]],
      [2.6, 2.6, 2.7, 2.7, 2.8, 2.8])
P("<b>Dinámico, Erlang y bloqueo.</b> Las conexiones llegan (Poisson, tasa λ) y se van (duración exponencial, media 1/μ) "
  "todo el tiempo. La carga en <b>Erlang</b> = λ/μ = número promedio de conexiones activas a la vez. Si no hay un bloque "
  "libre que cumpla continuidad y contigüidad, la solicitud se <b>bloquea</b>. La <b>probabilidad de bloqueo</b> "
  "(bloqueadas / totales) es nuestra métrica: más baja es mejor.")

# ============================================================================ 3. RL Y PPO
H1("3. Aprendizaje por refuerzo y PPO")
P("Un <b>agente</b> mira el estado del <b>entorno</b>, toma una <b>acción</b>, recibe una <b>recompensa</b> y el entorno "
  "cambia. Repitiendo esto miles de veces aprende una <b>política</b> que maximiza la recompensa acumulada. Es "
  "\"profundo\" (DRL) porque la política es una red neuronal.")
TABLE([["Concepto", "En nuestro problema"],
       ["Observación", "47 números: origen, destino, velocidad pedida y, por cada una de las 3 rutas: saltos, slots que "
        "necesita, espacio libre, dónde cabe y si cabe"],
       ["Acción", "Elegir la ruta 0, 1 o 2"],
       ["Recompensa", "+1 si la conexión se asignó, −1 si se bloqueó (maximizarla = minimizar el bloqueo)"],
       ["Paso / episodio", "Una solicitud / 1000 solicitudes (la red no se vacía entre episodios)"]],
      [3.4, 13.0])
P("<b>PPO</b> (<i>Proximal Policy Optimization</i>) es un algoritmo <b>actor-crítico</b>: el <i>actor</i> elige la ruta y el "
  "<i>crítico</i> estima qué tan buena es la situación, para saber si una acción fue mejor o peor de lo esperado. Recolecta "
  "2048 decisiones, actualiza las redes en grupos de <i>batch size</i> y <b>limita cuánto cambia la política</b> en cada "
  "actualización (de ahí \"Proximal\"), lo que lo hace estable. Usamos la implementación de Stable-Baselines3 con una red "
  "pequeña: 2 capas de 64 neuronas para el actor y otras 2 para el crítico (~15 000 parámetros).")
H2("3.1 Los tres hiperparámetros que optimizamos")
P("Un <b>hiperparámetro</b> es una perilla que se fija <i>antes</i> de entrenar (el agente no la aprende).")
B([f"<b>Learning rate</b>: qué tan grande es cada corrección de la red. Muy grande, oscila y olvida; muy chico, no "
   f"aprende. Por defecto 3·10⁻⁴; Optuna eligió <b>{sci(BP['learning_rate'])}</b>: aprende más lento pero más fino.",
   f"<b>Gamma</b>: cuánto le importa el futuro. El agente \"mira\" ~1/(1−γ) pasos adelante. Por defecto 0.99 (~100 "
   f"solicitudes); Optuna eligió <b>{BP['gamma']:.3f}</b> (~{1 / (1 - BP['gamma']):.0f} solicitudes).",
   f"<b>Batch size</b>: cuántas experiencias se usan en cada paso de gradiente. Por defecto 64; Optuna eligió "
   f"<b>{BP['batch_size']}</b>. " + ("Resultó el menos importante." if list(imp)[-1] == "batch_size" else
   "Con 100 trials parecía irrelevante; con 300 quedó segundo, pero lejos de gamma.")])
FIGURE(FIG / "gamma.png", 12, "Cuánto pesa una recompensa futura según gamma.")
P("<b>¿Por qué conviene mirar poco al futuro?</b> Las solicitudes llegan al azar: lo que pase en 100 solicitudes más "
  "depende casi nada de esta decisión y mucho del azar. Intentar predecirlo mete ruido en el aprendizaje. Con gamma ~0.84 "
  "el agente sigue considerando el efecto de corto plazo (no gastar espectro que la siguiente conexión va a necesitar) sin "
  "ese ruido. El resto de hiperparámetros de PPO quedó en sus valores por defecto.")

# ============================================================================ 4. DREAM-ON GYM
H1("4. DREAM-ON GYM")
P("DREAM-ON GYM (Jara, Pempelfort et al., SIMULTECH 2024) es un framework en Python para entrenar agentes de RL en redes "
  "ópticas. Tiene un <b>simulador de eventos discretos</b> (<i>simNetPy</i>: topología, slots, llegadas y salidas de "
  "conexiones, conteo de bloqueos) y un entorno Gymnasium que lo conecta con Stable-Baselines3.")
P("El simulador salta de evento en evento: en cada <b>llegada</b> llama a un <b>allocator</b>, una función que escribe el "
  "usuario y que decide cómo asignar la conexión; en cada <b>salida</b> libera los slots solo. Su firma es fija:")
CODE("""
def allocator(src, dst, b, c, n, path, action) -> (Controller.Status, Connection)
#  src, dst: origen y destino   b: velocidad pedida   c: la conexión a llenar
#  n: la red (estado de los slots)   path: las k rutas candidatas   action: lo que decidió el agente
""")
P("<b>Cómo lo usamos.</b> Usamos su simulador completo, su topología NSFNet y sus rutas. Tuvimos que resolver tres "
  "problemas de la versión publicada (0.0.9):")
B(["Declara TensorFlow, CUDA y MPI como dependencias, pero no los usa: lo instalamos sin ellos (~2 min en vez de horas).",
   "Al paquete le falta un archivo de esquema: lo instalamos desde el código fuente.",
   "Su entorno Gymnasium oficial está roto (error al crear eventos y observación constante): escribimos el nuestro, "
   "<i>RMLSAEnv</i>, sobre el simulador estándar, que sí funciona. Es la capa que el framework espera que el usuario "
   "defina de todos modos (observación, recompensa, fin de episodio)."])

# ============================================================================ 5. ESCENARIO
H1("5. Escenario, entorno y referencia")
FIGURE(FIG / "nsfnet.png", 14, "Topología NSFNet (disposición aproximada), con dos de las tres rutas candidatas entre 0 y 3.")
TABLE([["Parámetro", "Valor", "Por qué"],
       ["Topología", "NSFNet, 14 nodos, 21 enlaces, 100 slots", "Red estándar de la literatura; viene en DREAM-ON GYM"],
       ["Rutas candidatas", "k = 3 por par", "Suficiente para que haya decisiones reales"],
       ["Tráfico", "Poisson, duración exponencial (μ = 1)", "Modelo estándar de tráfico dinámico"],
       ["Velocidades", "10, 40, 100, 400 Gbps", "Mezcla de conexiones chicas y grandes (1000 Gbps se excluye: "
        "ocupa 80 de 100 slots y se bloquea siempre)"],
       ["Entrenamiento", "100 Erlang", "SPFF bloquea ~10 %: hay margen para mejorar"],
       ["Evaluación", f"{L(LOADS[0])} a {L(LOADS[-1])} Erlang ({len(LOADS)} cargas)", "Probar si lo aprendido generaliza"],
       ["Warm-up", "3000 solicitudes con SPFF", "Que la red parta cargada, como en operación real"]],
      [3.2, 5.6, 7.6])
P("<b>Semillas.</b> Todo lo aleatorio sale de generadores con semilla: misma semilla, mismo tráfico. Así dos agentes "
  "comparados con la misma semilla ven exactamente las mismas solicitudes. Usamos semillas <b>separadas</b> para cada "
  "cosa: 0 para entrenar, 1000–1002 para que Optuna mida cada trial y <b>2000–2009 para la prueba final</b>, que nadie vio "
  "antes. Evaluar con las semillas con que Optuna eligió inflaría los resultados (como dar una prueba con las preguntas "
  "del ensayo).")
H2("5.1 La referencia: SPFF")
P("<b>Shortest-Path First-Fit</b> usa siempre la ruta más corta y el primer bloque libre; si no cabe, bloquea. Es rápido "
  "y simple, pero no se adapta: aunque la ruta corta esté saturada y otra vacía, insiste con la corta. Nuestro agente "
  "usa <i>exactamente</i> la misma modulación y el mismo First-Fit; lo único distinto es la elección de ruta, así la "
  "comparación aísla el valor de esa decisión.")

# ============================================================================ 6. OPTUNA
H1("6. Optuna")
P("Optuna (Akiba et al., KDD 2019) optimiza hiperparámetros automáticamente. Se le entrega una <b>función objetivo</b> "
  "(\"con estos hiperparámetros, entrena y dime qué tan bien quedó\") y decide qué combinaciones probar y cuáles "
  "abandonar a medio camino.")
TABLE([["Concepto", "Qué es", "En nuestro proyecto"],
       ["Study / Trial", "La optimización completa / una prueba de una combinación", f"{N} trials; cada uno entrena un "
        "PPO 200 000 pasos a 100 Erlang"],
       ["Objective", "Lo que se minimiza", "Bloqueo con 10 000 solicitudes, promedio de 3 semillas (1000–1002)"],
       ["Sampler", "Cómo elige qué probar", "TPE (semilla 0, reproducible)"],
       ["Pruner", "Cuándo abandonar un trial", "MedianPruner"],
       ["Storage", "Dónde se guarda", "SQLite: si se corta, se retoma donde quedó"]],
      [2.8, 5.8, 7.8])
H2("6.1 TPE: cómo elige qué probar")
P("Los primeros 10 trials son aleatorios. Después, TPE separa los trials hechos en buenos y malos, estima dónde tienden a "
  "estar unos y otros, y propone el siguiente punto \"parecido a lo que funcionó, distinto a lo que falló\". Se nota: de "
  f"los {len(after)} trials posteriores a los 10 aleatorios, <b>{n_lr_low} usaron learning rate &lt; 2·10⁻⁴ y {n_g_low} "
  "gamma &lt; 0.9</b>, la zona buena.")
H2("6.2 MedianPruner: cómo ahorra tiempo")
P(f"Cada trial entrena en 5 etapas de 40 000 pasos y mide el bloqueo al final de cada una. Si va peor que la "
  f"<b>mediana</b> de los trials anteriores en esa etapa, se abandona (poda). Se podaron <b>{NP} de {N} trials</b>, lo "
  f"que ahorró ~{saved_min / 60:.1f} horas (sin poda el estudio habría tomado ~{(total_min + saved_min) / 60:.1f} h en "
  f"vez de {total_min / 60:.1f} h).")
H2("6.3 Espacio de búsqueda y función objetivo")
TABLE([["Hiperparámetro", "Rango", "Escala", "Por qué"],
       ["learning_rate", "10⁻⁵ a 10⁻²", "Logarítmica", "Los valores razonables abarcan 3 órdenes de magnitud"],
       ["gamma", "0.8 a 0.9999", "Lineal", "Desde horizonte corto (~5 pasos) hasta muy largo; incluye 0.99"],
       ["batch_size", "32, 64, 128, 256, 512", "Categórica", "Potencias de 2 que dividen el rollout de 2048; incluye 64"]],
      [3.0, 3.4, 2.4, 7.6])
CODE("""
def objective(trial):
    hp = {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
          "gamma":         trial.suggest_float("gamma", 0.8, 0.9999),
          "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])}
    model = PPO("MlpPolicy", RMLSAEnv(load=100, seed=0), seed=0, **hp)
    for stage in range(5):                                   # 5 etapas de 40 000 pasos
        model.learn(40_000, reset_num_timesteps=False)
        blocking = mean(evaluate_model(model, 100, s, 10_000) for s in (1000, 1001, 1002))
        trial.report(blocking, stage)                        # resultado parcial
        if trial.should_prune():                             # ¿peor que la mediana?
            raise optuna.TrialPruned()
    return blocking                                          # lo que Optuna minimiza
""")
H2("6.4 La evolución del estudio")
rows = [["Estudio", "Trials", "Pasos por trial", "Medición", "Mejor bloqueo", "Mejores hiperparámetros"],
        ["v1 (exploratorio)", "30", "50 000", "3 000 sol., 1 semilla", "0.0763", "γ 0.987, lr 4.8·10⁻⁴, batch 512"],
        ["v2", "40", "100 000", "10 000 sol., 1 semilla", "0.0726", "γ 0.830, lr 4.1·10⁻⁵, batch 256"],
        ["v3", "100", "200 000", "10 000 sol., 3 semillas", "0.0756", "γ 0.836, lr 2.2·10⁻⁵, batch 64"]]
if EXTENDED:
    rows.append([f"<b>Final ({N})</b>", f"<b>{N}</b>", "200 000", "10 000 sol., 3 semillas", f"<b>{f4(best.value)}</b>",
                 f"<b>γ {BP['gamma']:.3f}, lr {sci(BP['learning_rate'])}, batch {BP['batch_size']}</b>"])
TABLE(rows, [2.6, 1.4, 2.2, 3.4, 2.2, 4.6])
P("El v1 medía con pocas solicitudes y concluyó que convenía gamma alto; con mediciones más precisas (v2) la conclusión "
  "se invirtió. <b>Lección:</b> un estudio de hiperparámetros con mediciones ruidosas puede engañar. Desde el v3 cada trial "
  "se mide con 3 semillas para que ninguno gane por suerte." +
  (f" El estudio final extiende el v3 hasta {N} trials (sugerencia de la profesora) para comprobar si una búsqueda más "
   "larga encuentra algo mejor." if EXTENDED else ""))
if EXTENDED:
    same = best.number < 100
    P(("<b>Resultado de la extensión:</b> " + (
        f"con {N - 100} trials más, el mejor sigue siendo el trial {best.number} del estudio de 100. TPE siguió "
        "explorando y no encontró nada mejor: el óptimo ya se había encontrado." if same else
        f"el mejor pasó a ser el trial {best.number} (bloqueo {f4(best.value)}, contra 0.0756 con 100 trials), con "
        f"hiperparámetros en la misma zona: gamma {BP['gamma']:.3f} y learning rate {sci(BP['learning_rate'])}. La "
        "búsqueda más larga refinó el resultado sin cambiar la conclusión.")))

# ============================================================================ 7. RESULTADOS
H1("7. Resultados")
H2(f"7.1 Estudio Optuna ({N} trials)")
P(f"{N} trials en {total_min / 60:.1f} horas de cómputo: {NC} completos y {NP} podados. El mejor fue el trial "
  f"{best.number}, con bloqueo {f4(best.value)}. Los cinco mejores son prácticamente la misma receta:")
TABLE([["Trial", "Bloqueo", "gamma", "learning rate", "batch"]] +
      [[t.number, f4(t.value), f"{t.params['gamma']:.3f}", sci(t.params["learning_rate"]), t.params["batch_size"]]
       for t in top[:5]], [2.4, 3.0, 3.0, 4.0, 4.0])
g10 = [t.params["gamma"] for t in top10]; l10 = [t.params["learning_rate"] for t in top10]
FIGURE(OPT / "gamma_vs_learning_rate.png", 11, f"Dónde probó Optuna en el plano gamma–learning rate. Los mejores puntos "
       f"(amarillos) se agrupan en gamma {min(g10):.2f}–{max(g10):.2f} y learning rate {sci(min(l10))}–{sci(max(l10))}; "
       "las X son trials podados.")
FIGURE(OPT / "historial.png", 10.5, "Historial: cada punto es un trial completo; la línea roja, el mejor hasta ese momento.")
H2("7.2 Importancia de hiperparámetros")
FIGURE(OPT / "importancia.png", 10, "Importancia de cada hiperparámetro (método PED-ANOVA de Optuna).")
P("PED-ANOVA mira si los trials buenos tienen valores concentrados en una zona de cada hiperparámetro: si se concentran, "
  "ese hiperparámetro importa. <b>Ojo:</b> se calcula sobre los trials que se hicieron. Como TPE concentró casi todo en "
  "gamma bajo, dentro de esa zona lo que más separa a buenos de malos es el learning rate; por eso en estudios con menos "
  "trials (v2) gamma aparecía primero. La conclusión robusta de todos los estudios: <b>gamma y learning rate importan, el "
  "batch size no</b>." if not EXTENDED else
  "Con 100 trials el orden era learning rate 0.72, gamma 0.19; con 300 trials es " +
  ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in imp.items()) + ". El orden cambia según qué trials hay, "
  "porque TPE concentró casi toda la búsqueda en una zona (gamma bajo, learning rate bajo) y la importancia se mide "
  "dentro de lo explorado. Lo que se mantiene en todos los estudios es <b>dónde</b> están los buenos trials: "
  "<b>gamma entre 0.80 y 0.84 y learning rate cerca de 2·10⁻⁵</b>. <b>Ojo:</b> el mejor gamma "
  f"({BP['gamma']:.3f}) quedó en el borde inferior del rango buscado (0.8): un estudio futuro debería abrir el rango "
  "hacia abajo para ver si gamma aún menor sigue mejorando.")
H2("7.3 Comparación final")
P("3 agentes por defecto y 3 optimizados (semillas 0, 1 y 2), entrenados 300 000 pasos a 100 Erlang. Cada agente y SPFF "
  f"se evaluaron con 10 000 solicitudes en {len(LOADS)} cargas × 10 semillas nuevas. Para cada semilla se promedian los 3 "
  "entrenamientos, y con esos 10 valores se calcula la media y el <b>intervalo de confianza del 95 %</b>. La mejora se "
  "calcula <b>pareada</b>: en cada semilla se comparan los dos agentes sobre el mismo tráfico.")
FIGURE(FINAL / "bloqueo_vs_carga.png", 11, "Bloqueo vs carga (escala logarítmica) con IC 95 %. Las curvas no se cruzan.")
TABLE([["Carga", "SPFF", "PPO por defecto", "PPO Optuna", "Optuna vs SPFF", "Optuna vs defecto"]] +
      [[L(l), f"{f4(M.SPFF[l])} ± {f4(IC.SPFF[l])}", f"{f4(M['PPO defecto'][l])} ± {f4(IC['PPO defecto'][l])}",
        f"<b>{f4(M['PPO Optuna'][l])} ± {f4(IC['PPO Optuna'][l])}</b>",
        f"{imp_os['mean'][l]:.1f} ± {imp_os['ic95'][l]:.1f} %", f"{imp_od['mean'][l]:.1f} ± {imp_od['ic95'][l]:.1f} %"]
       for l in LOADS], [1.4, 2.9, 2.9, 3.1, 3.0, 3.1])
B([f"<b>Siempre el mismo orden</b> (SPFF > PPO por defecto > PPO Optuna en bloqueo). El optimizado gana a SPFF en "
   f"{wins_os}/{ncases} casos y al por defecto en {wins_od}/{ncases}. Los IC de las mejoras están lejos de cero.",
   "<b>Generaliza:</b> entrenó solo a 100 Erlang y mejora en todas las cargas.",
   f"<b>La mejora relativa baja con la carga</b> ({imp_os['mean'].iloc[0]:.0f} % → {imp_os['mean'].iloc[-1]:.0f} % vs "
   "SPFF): con la red muy llena ninguna ruta tiene espacio y elegir bien sirve menos.",
   f"<b>Consistencia:</b> a 100 Erlang los 3 optimizados están entre {f4(o100.min())} y {f4(o100.max())}; los por "
   f"defecto entre {f4(d100.min())} y {f4(d100.max())} ({(d100.max() - d100.min()) / (o100.max() - o100.min()):.0f} "
   "veces más dispersión). Con valores de fábrica el resultado depende de la suerte del entrenamiento.",
   f"<b>Matiz:</b> el peor de los 3 optimizados bloquea menos que el mejor de los 3 por defecto en {worst_beats_best} "
   f"de {ncases} casos: un entrenamiento por defecto con suerte se acerca, Optuna lo logra siempre."])
H2("7.4 Curvas de aprendizaje")
FIGURE(FINAL / "curvas_aprendizaje.png", 12, "Bloqueo por episodio durante el entrenamiento (media de 3 semillas; banda = "
       "mínimo y máximo).")
P(f"El optimizado baja de 10 % de bloqueo a los ~{np.mean(FB_O) / 1000:.0f} mil pasos; el por defecto tarda entre "
  f"{min(FB_D) / 1000:.0f} y {max(FB_D) / 1000:.0f} mil, de forma más irregular. Ambas curvas se aplanan antes de 150 000 "
  "pasos, con el optimizado siempre debajo: entrenar más no cambiaría la conclusión.")
H2("7.5 Cómo usa la red cada agente")
FIGURE(FINAL / "uso_rutas.png", 10.5, "Porcentaje de solicitudes enviadas por una ruta que no es la más corta. SPFF: 0 %.")
P(f"<b>La figura más reveladora.</b> El agente por defecto se desvía ~{PCT['PPO defecto'].mean():.0f} % de las veces "
  "<b>sin importar la carga</b>: aprendió que a veces conviene desviarse, pero no <i>cuándo</i>, y gasta espectro en rutas "
  f"largas sin necesidad. El optimizado se desvía {PCT['PPO Optuna'][LOADS[0]]:.0f} % con la red vacía y "
  f"{PCT['PPO Optuna'][LOADS[-1]]:.0f} % con la red saturada: un comportamiento <b>adaptativo</b> que nadie le programó.")
TABLE([["A 100 Erlang", "SPFF", "PPO por defecto", "PPO Optuna"],
       ["Bloqueo de 100 Gbps", f4(BR.SPFF[100]), f4(BR["PPO defecto"][100]), f4(BR["PPO Optuna"][100])],
       ["Bloqueo de 400 Gbps", f4(BR.SPFF[400]), f4(BR["PPO defecto"][400]), f"<b>{f4(BR['PPO Optuna'][400])}</b>"],
       ["Bloqueo de pares a > 3000 km", f4(DI.SPFF["> 3000"]), f4(DI["PPO defecto"]["> 3000"]),
        f"<b>{f4(DI['PPO Optuna']['> 3000'])}</b>"],
       ["Ocupación media del espectro", pc(OCC.SPFF * 100), pc(OCC["PPO defecto"] * 100), pc(OCC["PPO Optuna"] * 100)]],
      [5.2, 3.6, 3.8, 3.8])
P("Casi todo el bloqueo viene de las conexiones de <b>400 Gbps</b> (necesitan hasta 32 slots contiguos) y de los pares "
  "<b>lejanos</b> (rutas largas, más enlaces que coordinar y modulaciones que gastan más slots). Ahí es donde el optimizado "
  "gana más. Los dos PPO casi eliminan el bloqueo de 100 Gbps desviándolas cuando hace falta. El optimizado ocupa "
  "<b>más</b> espectro que SPFF: no es malo, significa que <b>acepta más conexiones</b> y aprovecha rutas que SPFF deja "
  "ociosas.")
FIGURE(FINAL / "mapa_ocupacion.png", 15.5, "Ocupación media de cada enlace a 100 Erlang: SPFF (izquierda) y PPO Optuna "
       "(derecha). La carga queda más repartida con el agente.")
H2("7.6 Maqueta en vivo")
FIGURE(FIG / "maqueta_corto.png", 15.5, "Un cuadro del video de la maqueta a 150 Erlang. Gris: ocupación de cada enlace. "
       "Rojo: solicitudes bloqueadas. Celeste: solicitudes que el agente aceptó por una ruta alternativa.")
P("<i>scripts/maqueta_red.py</i> hace que SPFF y el agente atiendan <b>la misma secuencia de solicitudes</b> y los dibuja "
  "lado a lado. Tiene un modo <b>video</b> (33 s, para la lámina: no puede fallar en vivo) y un modo <b>interactivo</b> con "
  "deslizadores de carga (25–250 Erlang) y velocidad, y un selector para comparar con el PPO por defecto. Para mostrar: a "
  "25 Erlang casi no hay rojo; a 250 hay mucho, pero el celeste del agente salva conexiones que SPFF pierde. Es una "
  "ilustración; las cifras oficiales son las de la sección 7.3.")

# ============================================================================ 8. RÁPIDO + LIMITACIONES
H1("8. Por qué fue rápido y qué limitaciones tiene")
P(f"Todo corre en un notebook sin GPU. El estudio de {N} trials tomó {total_min / 60:.1f} horas y la evaluación final "
  "~1 hora. Las razones:")
B(["<b>Instalación liviana:</b> DREAM-ON GYM no necesita TensorFlow ni CUDA, aunque los declare.",
   "<b>Formulación compacta:</b> el agente elige entre 3 rutas mirando 47 números (en trabajos como DeepRMSA también "
   "elige slots, con cientos de acciones y miles de valores de entrada).",
   "<b>Red pequeña</b> (~15 000 parámetros): en CPU cada actualización es instantánea; una GPU no ayudaría.",
   "<b>Recompensa en cada paso</b> (+1/−1): cada decisión tiene consecuencia inmediata y el agente aprende en ~30 000 pasos.",
   f"<b>Poda y TPE:</b> la poda ahorró ~{saved_min / 60:.1f} h; TPE evita recorrer una grilla (10 valores por eje "
   "serían 1 000 entrenamientos).",
   "<b>Bloqueos de 1–20 %</b>: 10 000 solicitudes bastan para medir con precisión.",
   "<b>Lo que sí tomó tiempo</b> fue diseñar el entorno y un protocolo de evaluación justo, no esperar al computador."])
H2("8.1 Limitaciones (decirlas antes de que las pregunten)")
B(["Una sola topología (NSFNet) y una sola carga de entrenamiento (100 Erlang).",
   "El agente solo elige la ruta; modulación y espectro son reglas fijas.",
   "Solo tres hiperparámetros; el resto de PPO quedó por defecto.",
   f"El mejor trial de Optuna es algo optimista (es el mejor de {N}): por eso la cifra oficial es la de la evaluación "
   f"con semillas nuevas ({f4(M['PPO Optuna'][100])} a 100 Erlang vs {f4(best.value)} en Optuna).",
   "Una sola heurística de referencia (SPFF), y todo es simulación, como en la literatura del área."])

# ============================================================================ 9. FAQ
H1("9. Preguntas probables")
QA("¿Optuna entrena al agente?", "No. Optuna propone configuraciones y decide cuáles abandonar; el que entrena es PPO "
   "dentro de cada trial. Optuna es el supervisor que prueba configuraciones; PPO, el alumno.")
QA("¿Qué parte de RMLSA decide el agente?", "La ruta (entre 3 candidatas). Modulación y espectro siguen las mismas reglas "
   "que SPFF, así la comparación aísla el valor de elegir bien la ruta.")
QA("¿Por qué le gana a SPFF si usa las mismas reglas?", "SPFF insiste en la ruta corta aunque esté llena. El agente ve "
   "cuánto espacio tiene cada ruta y si la conexión cabe, y aprende cuándo conviene desviarse.")
QA("¿Y por qué le gana al PPO por defecto, si los dos pueden desviarse?", f"El por defecto se desvía ~"
   f"{PCT['PPO defecto'].mean():.0f} % siempre, incluso cuando no hace falta, y gasta espectro que después falta. El "
   "optimizado se desvía de forma selectiva, según qué tan llena esté la red.")
QA("¿Por qué gamma tan bajo si lo normal es 0.99?", "0.99 es genérico, pensado para problemas donde las consecuencias "
   "llegan mucho después (juegos, robótica). Aquí las solicitudes son aleatorias e independientes: el efecto de una "
   "decisión se nota en las próximas pocas, no en las próximas cien.")
QA(f"¿Por qué {N} trials?", (f"Partimos con 100 y la profesora sugirió probar con ~300 para ver si una búsqueda más larga "
   f"encontraba algo mejor. Extendimos el mismo estudio hasta {N}: " + (
   "el mejor resultado no cambió, lo que confirma que el óptimo ya estaba encontrado." if best.number < 100 else
   "el resultado mejoró levemente, pero en la misma zona de hiperparámetros.")) if EXTENDED else
   "Con 3 hiperparámetros, 100 trials bastan para que TPE explore el espacio y converja; los mejores trials se repiten en "
   "la misma zona.")
QA("¿No es trampa evaluar con el mismo tráfico con que se eligió?", "No lo hacemos: Optuna midió con semillas 1000–1002 y "
   "la prueba final usa 2000–2009, con agentes nuevos entrenados desde cero con 3 semillas distintas.")
QA("¿Cómo saben que la diferencia no es azar?", f"Los IC 95 % de las mejoras están lejos de cero en todas las cargas; el "
   f"optimizado gana en {wins_od} de {ncases} casos al por defecto; y el resultado se repite con 3 entrenamientos y en "
   "estudios de Optuna distintos que llegan a los mismos hiperparámetros.")
QA("¿Qué es la poda y cuánto ayudó?", f"Abandonar un trial que va peor que la mediana. Se podaron {NP} de {N} trials, "
   f"ahorrando ~{saved_min / 60:.1f} horas.")
QA("¿Por qué no usaron el entorno oficial de DREAM-ON GYM?", "Está roto en la versión publicada (0.0.9). Usamos el "
   "simulador estándar del mismo framework y escribimos la capa Gymnasium, que el framework espera que defina el usuario.")
QA("¿Por qué no entrenaron más tiempo?", "Las curvas de aprendizaje se aplanan antes de 150 000 pasos y entrenamos "
   "300 000; el agente por defecto se estanca en un nivel peor.")
QA("¿Qué harían con más tiempo?", "Otras topologías, más hiperparámetros, otras heurísticas de referencia, y que el agente "
   "también elija modulación y espectro. Como extensión, conectarlo con un orquestador de red de más alto nivel (por "
   "ejemplo, uno multi-agéntico que reciba instrucciones en lenguaje natural y use al agente como herramienta).")

# ============================================================================ 10. GLOSARIO + REFERENCIAS
H1("10. Glosario y referencias")
gl = [("Batch size", "Experiencias usadas en cada paso de gradiente."),
      ("Bloqueo", "Rechazo de una solicitud por no encontrar espectro libre que cumpla continuidad y contigüidad."),
      ("DRL", "Aprendizaje por refuerzo profundo: la política es una red neuronal."),
      ("EON", "Red óptica elástica: espectro dividido en slots asignados según necesidad."),
      ("Erlang", "Carga: número promedio de conexiones activas a la vez (λ/μ)."),
      ("First-Fit", "Asignar el primer bloque de slots libres que sirva."),
      ("Gamma (γ)", "Factor de descuento: cuánto valen las recompensas futuras."),
      ("IC 95 %", "Intervalo de confianza: rango donde está el valor real con 95 % de confianza."),
      ("Learning rate", "Tamaño de cada corrección de los pesos de la red."),
      ("MedianPruner", "Regla de Optuna que abandona un trial si va peor que la mediana."),
      ("PED-ANOVA", "Método de Optuna para estimar la importancia de cada hiperparámetro."),
      ("PPO", "Algoritmo actor-crítico que limita cuánto cambia la política por actualización."),
      ("RMLSA", "Routing, Modulation Level and Spectrum Assignment."),
      ("Semilla", "Número que fija una secuencia aleatoria: misma semilla, mismos resultados."),
      ("SPFF", "Shortest-Path First-Fit: ruta más corta + First-Fit."),
      ("TPE", "Sampler de Optuna que aprende de los trials anteriores para elegir el siguiente.")]
TABLE([["Término", "Significado"]] + [[f"<b>{a}</b>", b_] for a, b_ in gl], [3.4, 13.0])
refs = ["X. Chen et al., \"DeepRMSA: A deep reinforcement learning framework for routing, modulation and spectrum "
        "assignment in elastic optical networks\", <i>J. Lightwave Technol.</i>, vol. 37, no. 16, 2019.",
        "N. Jara, H. Pempelfort et al., \"DREAM-ON GYM: A deep reinforcement learning environment for next-gen optical "
        "networks\", <i>SIMULTECH</i>, 2024.",
        "M. Doherty et al., \"Reinforcement learning for dynamic resource allocation in optical networks: Hype or hope?\", "
        "<i>J. Opt. Commun. Netw.</i>, vol. 17, 2025.",
        "T. Akiba et al., \"Optuna: A next-generation hyperparameter optimization framework\", <i>KDD</i>, 2019.",
        "J. Schulman et al., \"Proximal Policy Optimization Algorithms\", arXiv:1707.06347, 2017.",
        "A. Raffin et al., \"Stable-Baselines3: Reliable reinforcement learning implementations\", <i>JMLR</i>, vol. 22, 2021.",
        "J. Bergstra et al., \"Algorithms for hyper-parameter optimization\" (TPE), <i>NeurIPS</i>, 2011."]
P("<b>Referencias</b>", ParagraphStyle("rh", parent=body, spaceBefore=8))
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", ParagraphStyle("ref", parent=small, leftIndent=18, firstLineIndent=-18))

doc = Doc(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
          title="TEL317: Optuna + PPO + DREAM-ON GYM para RMLSA", author="Andrea Alfonzo, Nicolás Alvarez, Lorenzo Gonzalez")
doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
print("OK", OUT, "| estudio:", STUDY, "| trials:", N)
