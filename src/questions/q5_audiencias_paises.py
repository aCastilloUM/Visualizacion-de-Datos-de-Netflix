# -*- coding: utf-8 -*-
# Pregunta 5:
# ¿Qué países producen más contenido para audiencias adultas e infantiles?
#
# Buenas prácticas:
# - Reutiliza utils/cleaning.py para normalizar países y ratings, y mapear audiencias
# - Un gráfico por figura, sin 3D, paleta y estilo unificados (utils/plot_style.py)
# - Ejes legibles, ranking por segmentos (Top 1–10, 11–20, 21–30), nota de fuente
#
# Salidas:
# - outputs/q5/q5_top01_10_audiencias_pais.png
# - outputs/q5/q5_top11_20_audiencias_pais.png
# - outputs/q5/q5_top21_30_audiencias_pais.png
#

#  Pipeline Q5:
#    - Normaliza/expande países y ratings, mapea audiencias
#    - Rankea países por Total y arma 3 segmentos de Top
#    - Genera 3 gráficos agrupados
#    - Devuelve dict con DataFrames intermedios
#    

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps
from utils import cleaning as cl

# Preparación de datos

def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    
    # Normaliza/expande países -> country_final
    # Normaliza/explode ratings -> rating_norm
    # Mapea a audiencias: Adulto / Infantil
    
    if "country" not in df.columns or "rating" not in df.columns:
        raise ValueError("El DataFrame debe contener 'country' y 'rating'.")

    # Países (expande coproducciones y normaliza nombres)
    df_c = cl.expand_and_normalize_countries(df)

    # Ratings (normaliza alias y explode múltiples)
    df_r = cl.normalize_and_explode_ratings(df_c)

    # Audiencia (Adulto / Infantil)
    df_r["audiencia"] = df_r["rating_norm"].map(lambda x: cl.map_rating_to_audience(x, mode="adult_kids"))

    return df_r

def _pivot_country_audience(df_base: pd.DataFrame) -> pd.DataFrame:

    grp = df_base.groupby(["country_final", "audiencia"], as_index=False).size()
    pivot = grp.pivot(index="country_final", columns="audiencia", values="size").fillna(0).astype(int)

    for c in ("Infantil", "Adulto"):
        if c not in pivot.columns:
            pivot[c] = 0

    pivot["Total"] = pivot["Infantil"] + pivot["Adulto"]
    pivot = pivot.sort_values("Total", ascending=False)
    return pivot[["Infantil", "Adulto", "Total"]]

def _slice_top_segments(pivot_total: pd.DataFrame):

    top_1_10  = pivot_total.iloc[:10].copy().sort_values("Total", ascending=True)
    top_11_20 = pivot_total.iloc[10:20].copy().sort_values("Total", ascending=True)
    top_21_30 = pivot_total.iloc[20:30].copy().sort_values("Total", ascending=True)
    return top_1_10, top_11_20, top_21_30

# Plot
def _plot_grouped_barh(df_slice: pd.DataFrame, title: str, outpath: str):

    if df_slice.empty:
        return

    plt.figure(figsize=(12, max(5, 0.48 * len(df_slice))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    countries = df_slice.index.tolist()
    y = list(range(len(countries)))

    bar_h = 0.38
    ax.barh([i - bar_h/2 for i in y], df_slice["Infantil"], height=bar_h,
            label="Infantil", color=ps.COLOR_TV)
    ax.barh([i + bar_h/2 for i in y], df_slice["Adulto"], height=bar_h,
            label="Adulto", color=ps.COLOR_MOVIE)

    ax.set_yticks(y)
    ax.set_yticklabels(countries)
    ax.set_xlabel("Cantidad de títulos", color=ps.COLOR_TV)
    ax.set_title(title, fontsize=13, color=ps.COLOR_TV)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, min_n_ticks=6))
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.legend()

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:

    outdir_q5 = os.path.join(outdir, "q5")
    os.makedirs(outdir_q5, exist_ok=True)

    base  = _prepare_base(df)
    pivot = _pivot_country_audience(base)

    top_1_10, top_11_20, top_21_30 = _slice_top_segments(pivot)

    _plot_grouped_barh(top_1_10,  "Top 1–10 países (Adulto vs Infantil)",  os.path.join(outdir_q5, "q5_top01_10_audiencias_pais.png"))
    _plot_grouped_barh(top_11_20, "Top 11–20 países (Adulto vs Infantil)", os.path.join(outdir_q5, "q5_top11_20_audiencias_pais.png"))
    _plot_grouped_barh(top_21_30, "Top 21–30 países (Adulto vs Infantil)", os.path.join(outdir_q5, "q5_top21_30_audiencias_pais.png"))

    return {
        "base": base,
        "pivot": pivot,
        "top1_10": top_1_10,
        "top11_20": top_11_20,
        "top21_30": top_21_30,
    }
