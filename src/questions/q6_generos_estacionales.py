# Pregunta 6:
# ¿Se identifica alguna estacionalidad en los estrenos según la categoría (listed_in)?
# ¿Qué meses concentran más lanzamientos?
# Gráficos: heatmap (mes × categoría) + barras por mes (totales)
# Salidas: outputs/q6/q6_heatmap_categorias.png, outputs/q6/q6_barras_meses.png

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps
from utils import cleaning as cl

def _month_labels():
    return ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

#  Reindexa a meses 1..12. Soporta Series o DataFrames con index numérico de mes.

def _ensure_month_order(obj):

    return obj.reindex(range(1, 13), fill_value=0)

# Preparación
def _prepare_estacionalidad(df: pd.DataFrame) -> pd.DataFrame:

    dfx = cl.ensure_datetime(df, "date_added")
    dfx = dfx.dropna(subset=["date_added", "listed_in"]).copy()
    dfx["mes"] = dfx["date_added"].dt.month
    dfx = cl.explode_listed_in(dfx)  
    return dfx


# Plots

def _plot_heatmap_estacionalidad(df: pd.DataFrame, out_png_path: str):
    tabla = (
        df.pivot_table(index="mes", columns="listed_in", values="title", aggfunc="count", fill_value=0)
          .sort_index()
    )
    tabla = _ensure_month_order(tabla)

    plt.figure(figsize=(max(10, len(tabla.columns) * 0.35), 6), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    im = ax.imshow(tabla.values, aspect="auto")

    ax.set_yticks(range(12))
    ax.set_yticklabels(_month_labels())
    ax.set_xticks(range(len(tabla.columns)))
    ax.set_xticklabels(tabla.columns.tolist(), rotation=30, ha="right", fontsize=9)

    ax.set_title("Estacionalidad de estrenos por categoría (mes × categoría)", fontsize=13, color=ps.COLOR_TV)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("# estrenos")

    ps.add_source_note()  
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=220, facecolor=ps.COLOR_BG, bbox_inches="tight")
    plt.close()

def _plot_barras_totales_por_mes(df: pd.DataFrame, out_png_path: str):
    tot_mes = df.groupby("mes")["title"].count().sort_index()
    tot_mes = _ensure_month_order(tot_mes)

    plt.figure(figsize=(10, 5.5), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.bar(range(12), tot_mes.values, width=0.7, color=ps.COLOR_MOVIE, edgecolor=ps.COLOR_TV, linewidth=0.6)

    ax.set_xticks(range(12))
    ax.set_xticklabels(_month_labels())
    ax.set_ylabel("# estrenos", color=ps.COLOR_TV)
    ax.set_title("Cantidad total de estrenos por mes (todas las categorías)", fontsize=13, color=ps.COLOR_TV)

    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=6))

    ps.add_source_note()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=220, facecolor=ps.COLOR_BG, bbox_inches="tight")
    plt.close()

def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:
    outdir_q6 = os.path.join(outdir, "q6")
    os.makedirs(outdir_q6, exist_ok=True)

    base = _prepare_estacionalidad(df)

    tabla_mes_categoria = (
        base.pivot_table(index="mes", columns="listed_in", values="title", aggfunc="count", fill_value=0)
            .sort_index()
    )
    totales_mes = base.groupby("mes")["title"].count().sort_index()

    _plot_heatmap_estacionalidad(base, os.path.join(outdir_q6, "q6_heatmap_categorias.png"))
    _plot_barras_totales_por_mes(base, os.path.join(outdir_q6, "q6_barras_meses.png"))

    return {
        "base": base,
        "tabla_mes_categoria": tabla_mes_categoria,
        "totales_mes": totales_mes,
    }
