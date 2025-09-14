# -*- coding: utf-8 -*-
# Pregunta 2: ¿Cómo evolucionó la cantidad de estrenos a lo largo del tiempo para cada tipo de contenido?
# - Usa utils.cleaning.add_year_and_month para extraer year_added
# - Usa utils.plot_style para estilo y nota de fuente
# - Salidas: outputs/q2/q2_lineas_estrenos.png y outputs/q2/q2_area_apilada.png
# - Interfaz: run(df, outdir="outputs") -> devuelve pivot con columnas ['Movie','TV Show']

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps
from utils import cleaning as cl

def _aggregate_releases_by_year_and_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla por año con conteos de Movie y TV Show a partir de 'date_added'.
    """
    df2 = cl.add_year_and_month(df)  # añade year_added y month_added desde date_added
    grp = (
        df2.dropna(subset=["year_added", "type"])
           .groupby(["year_added", "type"], as_index=False)
           .size()
    )
    pivot = grp.pivot(index="year_added", columns="type", values="size").fillna(0).astype(int)
    pivot = pivot.sort_index()
    for col in ("Movie", "TV Show"):
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot[["Movie", "TV Show"]]

def _plot_lines(pivot: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(10, 5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.plot(pivot.index, pivot["Movie"],   label="Movies",   color=ps.COLOR_MOVIE, linewidth=2)
    ax.plot(pivot.index, pivot["TV Show"], label="TV Shows", color=ps.COLOR_TV,    linewidth=2)

    ax.set_title("Evolución de estrenos por año y tipo (date_added)", fontsize=13, color=ps.COLOR_TV)
    ax.set_xlabel("Año de agregado", color=ps.COLOR_TV)
    ax.set_ylabel("Cantidad de estrenos", color=ps.COLOR_TV)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    ps.add_source_note()  # "Fuente: Netflix dataset"
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def _plot_area_stacked(pivot: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(10, 5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.stackplot(
        pivot.index,
        pivot["Movie"],
        pivot["TV Show"],
        labels=["Movies", "TV Shows"],
        colors=[ps.COLOR_MOVIE, ps.COLOR_TV],
        alpha=0.9
    )

    ax.set_title("Composición de estrenos por año (área apilada)", fontsize=13, color=ps.COLOR_TV)
    ax.set_xlabel("Año de agregado", color=ps.COLOR_TV)
    ax.set_ylabel("Cantidad de estrenos", color=ps.COLOR_TV)
    ax.legend(loc="upper left")

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs") -> pd.DataFrame:
    outdir_q2 = os.path.join(outdir, "q2")
    os.makedirs(outdir_q2, exist_ok=True)

    pivot = _aggregate_releases_by_year_and_type(df)

    _plot_lines(pivot, os.path.join(outdir_q2, "q2_lineas_estrenos.png"))
    _plot_area_stacked(pivot, os.path.join(outdir_q2, "q2_area_apilada.png"))

    return pivot
