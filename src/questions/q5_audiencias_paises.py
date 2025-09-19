# Pregunta 5:
# ¿Qué países producen más contenido para audiencias Adulto/Infantil y Familiar/No Familiar?

# Pipeline:
# 1. Limpiar y normalizar países y ratings
# 2. Mapear ratings a audiencias (Adulto/Infantil, Familiar/No Familiar)
# 3. Calcular ranking de países por audiencia
# 4. Graficar los segmentos de países por audiencia

# Outputs:
# - outputs/q5/q5_top01_10_audiencias_pais.png
# - outputs/q5/q5_top11_20_audiencias_pais.png
# - outputs/q5/q5_top21_30_audiencias_pais.png

# Cleaning:
# - cl.expand_and_normalize_countries(df): Expande y normaliza los nombres de países para agrupar correctamente.
# - cl.normalize_and_explode_ratings(df): Normaliza los ratings y explota múltiples valores para mapear correctamente las audiencias.

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import plot_style as ps
from utils import cleaning as cl

FAMILIAR = {"TV-PG", "TV-G", "PG", "TV-Y", "TV-Y7"}
NO_FAMILIAR = {"TV-MA", "R", "NC-17", "PG-13", "NR", "TV-Y7-FV"}

def map_rating_to_familiar(rating: str) -> str:
    if rating in FAMILIAR:
        return "Familiar"
    elif rating in NO_FAMILIAR:
        return "No Familiar"
    else:
        return "No Familiar"


# Normaliza países y ratings, mapea a audiencias 
def _prepare_base(df: pd.DataFrame, mode: str = "adult_kids") -> pd.DataFrame:

    if "country" not in df.columns or "rating" not in df.columns:
        raise ValueError("El DataFrame debe contener 'country' y 'rating'.")

    
    df_c = cl.expand_and_normalize_countries(df)

    df_r = cl.normalize_and_explode_ratings(df_c)

    if mode == "adult_kids":
        df_r["audiencia"] = df_r["rating_norm"].map(
            lambda x: cl.map_rating_to_audience(x, mode="adult_kids")
        )
    elif mode == "familiar":
        df_r["audiencia"] = df_r["rating_norm"].map(map_rating_to_familiar)

    return df_r

# Agrupa por país y audiencia, calcula totales y ordena por Total desc
def _pivot_country_audience(df_base: pd.DataFrame) -> pd.DataFrame:
    grp = df_base.groupby(["country_final", "audiencia"], as_index=False).size()
    pivot = grp.pivot(index="country_final", columns="audiencia", values="size").fillna(0).astype(int)

    for c in df_base["audiencia"].unique():
        if c not in pivot.columns:
            pivot[c] = 0

    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    return pivot

def _slice_top_segments(pivot_total: pd.DataFrame):
    top_1_10  = pivot_total.iloc[:10].copy().sort_values("Total", ascending=True)
    top_11_20 = pivot_total.iloc[10:20].copy().sort_values("Total", ascending=True)
    top_21_30 = pivot_total.iloc[20:30].copy().sort_values("Total", ascending=True)
    return top_1_10, top_11_20, top_21_30

def _plot_grouped_barh(df_slice: pd.DataFrame, title: str, outpath: str):
    if df_slice.empty:
        return

    plt.figure(figsize=(12, max(5, 0.48 * len(df_slice))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    countries = df_slice.index.tolist()
    y = list(range(len(countries)))
    bar_h = 0.38

    cols = [c for c in df_slice.columns if c != "Total"]
    if len(cols) == 2:
        ax.barh([i - bar_h/2 for i in y], df_slice[cols[0]], height=bar_h,
                label=cols[0], color=ps.COLOR_TV)
        ax.barh([i + bar_h/2 for i in y], df_slice[cols[1]], height=bar_h,
                label=cols[1], color=ps.COLOR_MOVIE)

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

    results = {}

    # Adulto vs Infantil 
    base_adultkids = _prepare_base(df, mode="adult_kids")
    pivot_adultkids = _pivot_country_audience(base_adultkids)
    segs_adultkids = _slice_top_segments(pivot_adultkids)

    _plot_grouped_barh(segs_adultkids[0], "Top 1–10 países (Adulto vs Infantil)",
                       os.path.join(outdir_q5, "q5_top01_10_audiencias_adultkids.png"))
    _plot_grouped_barh(segs_adultkids[1], "Top 11–20 países (Adulto vs Infantil)",
                       os.path.join(outdir_q5, "q5_top11_20_audiencias_adultkids.png"))
    _plot_grouped_barh(segs_adultkids[2], "Top 21–30 países (Adulto vs Infantil)",
                       os.path.join(outdir_q5, "q5_top21_30_audiencias_adultkids.png"))

    results["adultkids"] = {
        "base": base_adultkids,
        "pivot": pivot_adultkids,
        "segments": segs_adultkids,
    }

    # Familiar vs No Familiar 
    base_familiar = _prepare_base(df, mode="familiar")
    pivot_familiar = _pivot_country_audience(base_familiar)
    segs_familiar = _slice_top_segments(pivot_familiar)

    _plot_grouped_barh(segs_familiar[0], "Top 1–10 países (Familiar vs No Familiar)",
                       os.path.join(outdir_q5, "q5_top01_10_audiencias_familiar.png"))
    _plot_grouped_barh(segs_familiar[1], "Top 11–20 países (Familiar vs No Familiar)",
                       os.path.join(outdir_q5, "q5_top11_20_audiencias_familiar.png"))
    _plot_grouped_barh(segs_familiar[2], "Top 21–30 países (Familiar vs No Familiar)",
                       os.path.join(outdir_q5, "q5_top21_30_audiencias_familiar.png"))

    results["familiar"] = {
        "base": base_familiar,
        "pivot": pivot_familiar,
        "segments": segs_familiar,
    }

    results.update({
        "base":  base_adultkids,
        "pivot": pivot_adultkids,
        "top1_10":  segs_adultkids[0],
        "top11_20": segs_adultkids[1],
        "top21_30": segs_adultkids[2],
    })

    return results
