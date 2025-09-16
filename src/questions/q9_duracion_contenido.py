# -*- coding: utf-8 -*-
# Pregunta 9: ¿Cuál es la distribución en duración en series y películas?

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_style as ps
from utils import cleaning as cl

def _plot_hist_movies(df: pd.DataFrame, outpath: str):
    data = df["duration_minutes"].dropna()
    if data.empty:
        return
    plt.figure(figsize=(10, 5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)
    # Usar más bins para mayor detalle en el eje X
    ax.hist(data, bins=60, color=ps.COLOR_MOVIE, edgecolor=ps.COLOR_TV, alpha=0.85)
    ax.set_xlabel("Duración (minutos)", color=ps.COLOR_TV)
    ax.set_ylabel("# Películas", color=ps.COLOR_TV)
    ax.set_title("Distribución de duración de películas", fontsize=13, color=ps.COLOR_TV)
    ps.add_source_note("Fuente: Netflix dataset. Elaboración propia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def _plot_hist_tvshows(df: pd.DataFrame, outpath: str):
    data = df["duration_seasons"].dropna()
    if data.empty:
        return
    # Usar bins más pequeños para mayor detalle en el eje X
    bins = np.arange(1, data.max() + 2, 1)  # medio entero para más valores
    plt.figure(figsize=(9, 5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)
    ax.hist(data, bins=bins, color=ps.COLOR_MOVIE, edgecolor=ps.COLOR_MOVIE, alpha=0.85)
    ax.set_xlabel("Cantidad de temporadas", color=ps.COLOR_TV)
    ax.set_ylabel("# Series", color=ps.COLOR_TV)
    ax.set_title("Distribución de duración de series (temporadas)", fontsize=13, color=ps.COLOR_TV)
    ax.set_xticks(np.arange(1, int(data.max()) + 1, 1))

    ps.add_source_note("Fuente: Netflix dataset. Elaboración propia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:
    outdir_q9 = os.path.join(outdir, "q9")
    os.makedirs(outdir_q9, exist_ok=True)

    df_clean = cl.normalize_duration(df)

    movies = df_clean[df_clean["type"] == "Movie"].copy()
    tvshows = df_clean[df_clean["type"] == "TV Show"].copy()

    _plot_hist_movies(movies, os.path.join(outdir_q9, "q9_movies_duration_hist.png"))
    _plot_hist_tvshows(tvshows, os.path.join(outdir_q9, "q9_tvshows_duration_hist.png"))

    return {
        "movies": movies[["title", "duration_minutes"]].dropna(),
        "tvshows": tvshows[["title", "duration_seasons"]].dropna(),
    }
