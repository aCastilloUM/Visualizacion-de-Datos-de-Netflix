# Pregunta 10:
# ¿Hay palabras que se utilicen más que otras en títulos y descripciones?
#
# - Limpieza/tokenización EN: utils.cleaning.count_top_words
# - Gráficos: 2 barh (Top 20) para títulos y para descripciones
#
# Salidas:
# - outputs/q10/q10_top_words_titles.png
# - outputs/q10/q10_top_words_descriptions.png
#
# Pipeline Q10:
# Cuenta palabras más frecuentes en títulos (title) y descripciones (description)
# Genera 2 barh con Top-N
# Devuelve dict con las Series de frecuencias

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps
from utils import cleaning as cl

def _plot_top_words_barh(freqs: pd.Series, title: str, color: str, outpath: str):
    if freqs is None or freqs.empty:
        return

    plt.figure(figsize=(12, max(5, 0.5 * len(freqs))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.barh(freqs.index.tolist(), freqs.values.tolist(),
            color=color, edgecolor=ps.COLOR_TV, linewidth=0.6)

    ax.set_xlabel("Frecuencia", color=ps.COLOR_TV)
    ax.set_title(title, fontsize=13, color=ps.COLOR_TV)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=6))

    ps.add_source_note()  
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG, bbox_inches="tight")
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs", topn: int = 20) -> dict:

    outdir_q10 = os.path.join(outdir, "q10")
    os.makedirs(outdir_q10, exist_ok=True)

    if "title" not in df.columns or "description" not in df.columns:
        raise ValueError("El DataFrame debe contener 'title' y 'description'.")

    # Top palabras (EN) en títulos y descripciones
    top_titles = cl.count_top_words(df["title"],   topn=topn, min_len=3)
    top_desc   = cl.count_top_words(df["description"], topn=topn, min_len=3)

    # Gráficos (usamos colores distintos para diferenciarlos)
    _plot_top_words_barh(top_titles, "Top palabras en títulos", ps.COLOR_MOVIE,
                         os.path.join(outdir_q10, "q10_top_words_titles.png"))
    _plot_top_words_barh(top_desc,   "Top palabras en descripciones", ps.COLOR_TV,
                         os.path.join(outdir_q10, "q10_top_words_descriptions.png"))

    return {
        "top_words_titles": top_titles,       
        "top_words_descriptions": top_desc,
    }
