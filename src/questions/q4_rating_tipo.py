# -*- coding: utf-8 -*-
# Pregunta 4:
# ¿Qué tipo de contenido es más común para cada rating?
# - Reutiliza utils.cleaning.normalize_and_explode_ratings
# - Gráficos: barras agrupadas (conteos) + barras apiladas 100% (proporciones)
# - Salidas: outputs/q4/q4_rating_tipo_grouped_barh.png, outputs/q4/q4_rating_tipo_stacked100.png
# - Interfaz: run(df, outdir="outputs") -> {"counts": pivot_counts, "props": pivot_props}

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps
from utils import cleaning as cl

def _prepare_ratings(df: pd.DataFrame) -> pd.DataFrame:

    if "rating" not in df.columns or "type" not in df.columns:
        raise ValueError("El DataFrame debe contener 'rating' y 'type'.")

    dfx = df.dropna(subset=["rating", "type"]).copy()
    dfx["type"] = dfx["type"].astype(str).str.strip()
    dfx = cl.normalize_and_explode_ratings(dfx)  # agrega rating_norm
    return dfx

def _pivot_counts(dfx: pd.DataFrame) -> pd.DataFrame:
    grp = dfx.groupby(["rating_norm", "type"], as_index=False).size()
    pivot = grp.pivot(index="rating_norm", columns="type", values="size").fillna(0).astype(int)

    for col in ("Movie", "TV Show"):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Total"] = pivot["Movie"] + pivot["TV Show"]
    pivot = pivot.sort_values("Total", ascending=False)
    return pivot[["Movie", "TV Show", "Total"]]

def _pivot_props(pivot_counts: pd.DataFrame) -> pd.DataFrame:
    pc = pivot_counts[pivot_counts["Total"] > 0].copy()
    pc["Movie_pct"] = pc["Movie"] / pc["Total"]
    pc["TV_pct"]    = pc["TV Show"] / pc["Total"]
    return pc[["Movie_pct", "TV_pct", "Total"]]

# Plots 

def _plot_grouped_barh_counts(pivot_counts: pd.DataFrame, outpath: str) -> None:
    if pivot_counts.empty:
        return

    df_plot = pivot_counts.sort_values("Total", ascending=True)

    plt.figure(figsize=(12, max(5, 0.45 * len(df_plot))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ratings = df_plot.index.tolist()
    y = list(range(len(ratings)))
    bar_h = 0.38

    ax.barh([i - bar_h/2 for i in y], df_plot["Movie"],   height=bar_h, label="Movies",   color=ps.COLOR_MOVIE)
    ax.barh([i + bar_h/2 for i in y], df_plot["TV Show"], height=bar_h, label="TV Shows", color=ps.COLOR_TV)

    ax.set_yticks(y)
    ax.set_yticklabels(ratings)
    ax.set_xlabel("Cantidad de títulos", color=ps.COLOR_TV)
    ax.set_title("Conteo por rating y tipo de contenido", fontsize=13, color=ps.COLOR_TV)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=6))
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend()

    ps.add_source_note()  
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def _plot_stacked_100_props(pivot_props: pd.DataFrame, outpath: str) -> None:
    if pivot_props.empty:
        return

    df_plot = pivot_props.copy()
    order = df_plot["Total"].sort_values(ascending=True).index
    df_plot = df_plot.loc[order]

    plt.figure(figsize=(12, max(5, 0.45 * len(df_plot))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ratings = df_plot.index.tolist()
    y = list(range(len(ratings)))
    bar_h = 0.6

    left = [0.0] * len(df_plot)
    ax.barh(y, df_plot["Movie_pct"].values, height=bar_h, left=left, color=ps.COLOR_MOVIE, label="Movie")
    left = df_plot["Movie_pct"].values
    ax.barh(y, df_plot["TV_pct"].values,    height=bar_h, left=left, color=ps.COLOR_TV,    label="TV Show")

    ax.set_yticks(y)
    ax.set_yticklabels(ratings)
    ax.set_xlabel("Proporción", color=ps.COLOR_TV)
    ax.set_title("Proporción por rating (100%)", fontsize=13, color=ps.COLOR_TV)

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([f"{int(v*100)}%" for v in [0, 0.25, 0.5, 0.75, 1.0]])

    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(loc="lower right")

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()


def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:
    outdir_q4 = os.path.join(outdir, "q4")
    os.makedirs(outdir_q4, exist_ok=True)

    base = _prepare_ratings(df)
    pivot_counts = _pivot_counts(base)
    pivot_props  = _pivot_props(pivot_counts)

    _plot_grouped_barh_counts(pivot_counts, os.path.join(outdir_q4, "q4_rating_tipo_grouped_barh.png"))
    _plot_stacked_100_props(pivot_props,   os.path.join(outdir_q4, "q4_rating_tipo_stacked100.png"))

    return {"counts": pivot_counts, "props": pivot_props}
