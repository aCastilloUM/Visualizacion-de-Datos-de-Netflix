# Pregunta 1:
# ¿Cómo ha cambiado la proporción entre películas y series a lo largo de los años?

# Pipeline:
# 1. Calcular proporciones por año (release_year) y tipo
# 2. Generar PNG en outputs/q1/q1_proportion_movies_series.png
# 3. Devolver el DataFrame de proporciones 

# Outputs:
# - outputs/q1/q1_proportion_movies_series.png

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
from utils import plot_style as ps


# Calcula la proporción de películas/series por release_year, devuelve DataFrame con columnas: Movie, TV Show, total, prop_movies, prop_series
def calculate_proportion(df):
    if "release_year" not in df.columns or "type" not in df.columns:
        raise ValueError("DataFrame must contain 'release_year' and 'type'.")

    dfx = df.copy()
    dfx["type"] = dfx["type"].astype(str).str.strip()
    dfx = dfx.dropna(subset=["release_year", "type"])
    dfx["release_year"] = dfx["release_year"].astype(int)

    counts = (
        dfx.groupby(["release_year", "type"])
           .size()
           .unstack(fill_value=0)
           .sort_index()
    )
    for col in ("Movie", "TV Show"):
        if col not in counts.columns:
            counts[col] = 0

    counts["total"] = counts["Movie"] + counts["TV Show"]
    counts = counts[counts["total"] > 0].copy()
    counts["prop_movies"] = counts["Movie"] / counts["total"]
    counts["prop_series"] = counts["TV Show"] / counts["total"]
    return counts[["Movie", "TV Show", "total", "prop_movies", "prop_series"]]


def plot_proportion(counts, outpath):
    if counts.empty:
        return

    plt.figure(figsize=(10, 6), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.plot(counts.index, counts["prop_movies"], label="Movies", linewidth=2, color=ps.COLOR_MOVIE)
    ax.plot(counts.index, counts["prop_series"], label="Series", linewidth=2, color=ps.COLOR_TV)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Release year", color=ps.COLOR_TV)
    ax.set_ylabel("Proportion", color=ps.COLOR_TV)
    ax.set_title("Proportion of movies and series by year", fontsize=13, color=ps.COLOR_TV)

    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.legend(title="Content type")

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()


def run(df, outdir="outputs"):
    outdir_q1 = os.path.join(outdir, "q1")
    os.makedirs(outdir_q1, exist_ok=True)

    counts = calculate_proportion(df)
    plot_proportion(counts, os.path.join(outdir_q1, "q1_proportion_movies_series.png"))
    return counts
