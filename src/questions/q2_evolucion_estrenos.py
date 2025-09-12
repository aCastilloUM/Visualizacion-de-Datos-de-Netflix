# -*- coding: utf-8 -*-
# Pregunta 2: ¿Cómo evolucionó la cantidad de estrenos a lo largo del tiempo para cada tipo de contenido?
# Reglas de visualización:
# - Matplotlib (sin seaborn)
# - Un gráfico por figura
# - No fijar colores específicos (si luego querés paleta, la agregamos)

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def _parse_year_from_date_added(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte date_added a datetime y extrae year_added (sin modificar df original)."""
    df2 = df.copy()
    df2["date_added"] = pd.to_datetime(df2["date_added"], errors="coerce")
    df2["year_added"] = df2["date_added"].dt.year
    return df2

def _aggregate_releases_by_year_and_type(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por year_added y type -> tabla pivot con columnas 'Movie' y 'TV Show'."""
    grp = (
        df.dropna(subset=["year_added"])
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
    """Gráfico de líneas comparativo (Movies vs TV Shows)."""
    plt.figure(figsize=(10, 5))
    plt.plot(pivot.index, pivot["Movie"], label="Movies")
    plt.plot(pivot.index, pivot["TV Show"], label="TV Shows")
    plt.title("Evolución de estrenos por año y tipo (date_added)")
    plt.xlabel("Año de agregado")
    plt.ylabel("Cantidad de estrenos")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _plot_area_stacked(pivot: pd.DataFrame, outpath: str) -> None:
    """Gráfico de área apilada para composición en el tiempo."""
    plt.figure(figsize=(10, 5))
    plt.stackplot(pivot.index, pivot["Movie"], pivot["TV Show"], labels=["Movies", "TV Shows"])
    plt.title("Composición de estrenos por año (área apilada)")
    plt.xlabel("Año de agregado")
    plt.ylabel("Cantidad de estrenos")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs") -> pd.DataFrame:
    """
    Pipeline de la Pregunta 2:
    - recibe el DataFrame ya leído
    - limpia/deriva year_added
    - agrega estrenos por año y tipo
    - exporta CSV + PNGs
    - devuelve el DataFrame agregado (pivot)
    """
    os.makedirs(outdir, exist_ok=True)

    df2 = _parse_year_from_date_added(df)
    pivot = _aggregate_releases_by_year_and_type(df2)

    # Exportar tabla
    csv_out = os.path.join(outdir, "q2_estrenos_por_anio.csv")
    pivot.to_csv(csv_out, index_label="year_added")

    # Gráficos (un gráfico por figura)
    line_out = os.path.join(outdir, "q2_lineas_estrenos.png")
    _plot_lines(pivot, line_out)

    area_out = os.path.join(outdir, "q2_area_apilada.png")
    _plot_area_stacked(pivot, area_out)

    return pivot
