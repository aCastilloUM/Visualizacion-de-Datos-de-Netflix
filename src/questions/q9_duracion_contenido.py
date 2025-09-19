# Pregunta 9:
# ¿Cuál es la distribución en duración en series y películas?

# Pipeline:
# 1. Limpiar y normalizar la columna 'duration' con normalize_duration
# 2. Separar películas y series por tipo
# 3. Graficar histogramas de duración para películas y series
# 4. Devolver los DataFrames filtrados por tipo y duración

# Outputs:
# - outputs/q9/q9_movies_duration_hist.png 
# - outputs/q9/q9_tvshows_duration_hist.png

# Cleaning:
# - cl.normalize_duration(df): Normaliza la columna 'duration' y la separa en minutos para películas y temporadas para series.

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_style as ps
from utils import cleaning as cl


# Añade líneas de media y mediana, con etiquetas
def _add_central_tendency_lines(ax, mean_val: float, median_val: float, y_top: float):
    ax.axvline(mean_val, color="#000000", linestyle="--", linewidth=1.8)
    ax.axvline(median_val, color="#00ff00", linestyle="--", linewidth=1.8)

    # Chequear distancia
    if abs(mean_val - median_val) < 2:
        ax.text(mean_val, y_top * 1.05, "Media", color="#000000",
                ha="center", va="bottom", fontsize=9)
        ax.text(median_val, y_top * 1.10, "Mediana", color="#00ff00",
                ha="center", va="bottom", fontsize=9)
    else:
        ax.text(mean_val + 0.5, y_top * 1.02, "Media", color="#000000",
                ha="left", va="bottom", fontsize=9)
        ax.text(median_val - 1, y_top * 1.02, "Mediana", color="#00ff00",
                ha="right", va="bottom", fontsize=9)



# Calcula media y mediana, dict
def _compute_basic_stats(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if s.empty:
        return {}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
    }


def _plot_hist_movies(df: pd.DataFrame, outpath: str) -> dict:
    data = df["duration_minutes"].dropna().astype(float)
    if data.empty:
        return {}

    stats = _compute_basic_stats(data)

    plt.figure(figsize=(12, 6), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    n_bins = 60

    counts, bins, patches = ax.hist(
        data,
        bins=n_bins,
        color=ps.COLOR_MOVIE,
        edgecolor="#000000",   
        alpha=0.9
    )

    # Etiquetas arriba de cada barra
    for rect, count in zip(patches, counts):
        if count > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{int(count)}", ha="center", va="bottom", fontsize=8, color="#222")

    ax.set_xlabel("Duración (minutos)", color=ps.COLOR_TV)
    ax.set_ylabel("# Películas", color=ps.COLOR_TV)
    ax.set_title("Distribución de duración de películas", fontsize=13, color=ps.COLOR_TV)

    # Líneas de media y mediana
    _add_central_tendency_lines(ax, stats["mean"], stats["median"], counts.max())

    ps.add_source_note("Fuente: Netflix dataset. Elaboración propia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

    return stats


def _plot_hist_tvshows(df: pd.DataFrame, outpath: str) -> dict:
    data = df["duration_seasons"].dropna().astype(float)
    if data.empty:
        return {}

    stats = _compute_basic_stats(data)
    max_seasons = int(data.max())
    bins = np.arange(0.5, max_seasons + 1.5, 1)

    plt.figure(figsize=(12, 6), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)


    counts, bins, patches = ax.hist(
        data,
        bins=bins,
        color=ps.COLOR_MOVIE,
        edgecolor="#000000",   
        alpha=0.9
    )

    # Etiquetas arriba de cada barra
    for rect, count in zip(patches, counts):
        if count > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{int(count)}", ha="center", va="bottom", fontsize=8, color="#222")

    ax.set_xlabel("Cantidad de temporadas", color=ps.COLOR_TV)
    ax.set_ylabel("# Series", color=ps.COLOR_TV)
    ax.set_title("Distribución de duración de series (temporadas)", fontsize=13, color=ps.COLOR_TV)
    ax.set_xticks(np.arange(1, max_seasons + 1, 1))

    _add_central_tendency_lines(ax, stats["mean"], stats["median"], counts.max())

    ps.add_source_note("Fuente: Netflix dataset. Elaboración propia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

    return stats


def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:
    outdir_q9 = os.path.join(outdir, "q9")
    os.makedirs(outdir_q9, exist_ok=True)

    df_clean = cl.normalize_duration(df)

    movies = df_clean[df_clean["type"] == "Movie"].copy()
    tvshows = df_clean[df_clean["type"] == "TV Show"].copy()

    stats_movies = _plot_hist_movies(movies, os.path.join(outdir_q9, "q9_movies_duration_hist.png"))
    stats_tv     = _plot_hist_tvshows(tvshows, os.path.join(outdir_q9, "q9_tvshows_duration_hist.png"))

    return {
        "movies": movies[["title", "duration_minutes"]].dropna(),
        "tvshows": tvshows[["title", "duration_seasons"]].dropna(),
        "stats": {
            "movies": stats_movies,
            "tvshows": stats_tv,
        }
    }
