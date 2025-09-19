# Pregunta 10:
# ¿Hay palabras que se utilicen más que otras en títulos y descripciones?
#
# Pipeline:
# 1. Limpiar y tokenizar textos con count_top_words
# 2. Calcular las palabras más frecuentes en títulos y descripciones
# 3. Graficar los Top-N en dos gráficos de barras horizontales
# 4. Devolver un diccionario con las Series de frecuencias
#
# Outputs:
# - outputs/q10/q10_top_words_titles.png
# - outputs/q10/q10_top_words_descriptions.png
#
# Cleaning:
# - cl.count_top_words(series): Tokeniza y cuenta las palabras más frecuentes en una serie de textos en inglés, filtrando palabras cortas y stopwords.

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import plot_style as ps
from utils import cleaning as cl

# Gráfico de barras horizontales para frecuencias de palabras
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

    top_titles = cl.count_top_words(df["title"],   topn=topn, min_len=3)
    top_desc   = cl.count_top_words(df["description"], topn=topn, min_len=3)

    color_rojo = "red"
    _plot_top_words_barh(top_titles, "Top palabras en títulos", color_rojo,
                         os.path.join(outdir_q10, "q10_top_words_titles.png"))
    _plot_top_words_barh(top_desc,   "Top palabras en descripciones", color_rojo,
                         os.path.join(outdir_q10, "q10_top_words_descriptions.png"))

    return {
        "top_words_titles": top_titles,       
        "top_words_descriptions": top_desc,
    }
