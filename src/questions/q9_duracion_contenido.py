# Pregunta 9:
# ¿Cuál es la distribución en duración en series y películas?
#
# Buenas prácticas:
# - Reutiliza utils.cleaning.normalize_duration
# - Gráficos separados para Movies (minutos) y TV Shows (temporadas)
# - Paleta y fuente con utils.plot_style
#
# Salidas:
# - outputs/q9/q9_movies_duration_hist.png
# - outputs/q9/q9_tvshows_duration_hist.png

#   Pipeline Q9:
#   Normaliza duración en minutos/temporadas
#   Genera histogramas para Movies y TV Shows
#   Devuelve dict con DataFrames filtrados


from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_style as ps
from utils import cleaning as cl

# Plot
def _plot_hist_movies(df: pd.DataFrame, outpath: str):
    data = df["duration_minutes"].dropna()
    if data.empty:
        return

    plt.figure(figsize=(10, 5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.hist(data, bins=30, color=ps.COLOR_MOVIE, edgecolor=ps.COLOR_TV, alpha=0.85)

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

    plt.figure(figsize=(8, 5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.hist(data, bins=range(1, int(data.max()) + 2), color=ps.COLOR_TV, edgecolor=ps.COLOR_MOVIE, alpha=0.85)

    ax.set_xlabel("Cantidad de temporadas", color=ps.COLOR_TV)
    ax.set_ylabel("# Series", color=ps.COLOR_TV)
    ax.set_title("Distribución de duración de series (temporadas)", fontsize=13, color=ps.COLOR_TV)

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
