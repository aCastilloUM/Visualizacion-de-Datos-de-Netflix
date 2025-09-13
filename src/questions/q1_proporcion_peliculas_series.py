# -*- coding: utf-8 -*-
# Pregunta 1:
# ¿Cómo ha cambiado la proporción entre películas y series a lo largo de los años?
# - Usa utils/plot_style.py (paleta + fuente)
# - Sólo genera PNG en outputs/q1/
# - Interfaz: run(df, outdir="outputs") -> devuelve DataFrame con proporciones

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from utils import plot_style as ps

def _calcular_proporcion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula proporción de películas/series por release_year.
    Retorna DataFrame con columnas: Movie, TV Show, total, prop_peliculas, prop_series
    """
    if "release_year" not in df.columns or "type" not in df.columns:
        raise ValueError("El DataFrame debe contener 'release_year' y 'type'.")

    dfx = df.copy()
    # Normalización defensiva
    dfx["type"] = dfx["type"].astype(str).str.strip()
    dfx = dfx.dropna(subset=["release_year", "type"])
    dfx["release_year"] = dfx["release_year"].astype(int)

    conteo = (
        dfx.groupby(["release_year", "type"])
           .size()
           .unstack(fill_value=0)
           .sort_index()
    )
    # Garantizar columnas
    for col in ("Movie", "TV Show"):
        if col not in conteo.columns:
            conteo[col] = 0

    conteo["total"] = conteo["Movie"] + conteo["TV Show"]
    # Evitar división por cero
    conteo = conteo[conteo["total"] > 0].copy()
    conteo["prop_peliculas"] = conteo["Movie"] / conteo["total"]
    conteo["prop_series"]   = conteo["TV Show"] / conteo["total"]
    return conteo[["Movie", "TV Show", "total", "prop_peliculas", "prop_series"]]

def _graficar_proporcion(conteo: pd.DataFrame, outpath: str) -> None:
    """
    Gráfico de líneas de proporciones por año: Películas vs Series.
    """
    if conteo.empty:
        return

    plt.figure(figsize=(10, 6), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    # Series en español, colores consistentes con el resto del proyecto
    ax.plot(conteo.index, conteo["prop_peliculas"], label="Películas", linewidth=2, color=ps.COLOR_MOVIE)
    ax.plot(conteo.index, conteo["prop_series"],   label="Series",    linewidth=2, color=ps.COLOR_TV)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_yticklabels([f"{int(y*100)}%" for y in ax.get_yticks()])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Año de lanzamiento", color=ps.COLOR_TV)
    ax.set_ylabel("Proporción", color=ps.COLOR_TV)
    ax.set_title("Proporción de películas y series por año", fontsize=13, color=ps.COLOR_TV)

    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.legend(title="Tipo de contenido")

    ps.add_source_note("Fuente: Netflix dataset. Elaboración propia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs") -> pd.DataFrame:
    """
    Pipeline Q1:
    - Calcula proporciones por año (release_year) y type
    - Genera PNG en outputs/q1/q1_proporcion_peliculas_series.png
    - Devuelve el DataFrame de proporciones (para inspección/test)
    """
    outdir_q1 = os.path.join(outdir, "q1")
    os.makedirs(outdir_q1, exist_ok=True)

    conteo = _calcular_proporcion(df)
    _graficar_proporcion(conteo, os.path.join(outdir_q1, "q1_proporcion_peliculas_series.png"))
    return conteo
