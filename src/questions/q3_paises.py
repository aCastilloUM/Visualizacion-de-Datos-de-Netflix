# Pregunta 3:
# ¿Qué países producen más contenido? ¿Cómo se distribuye por tipo (Movies vs TV Shows)?

# Pipeline:
# 1) Limpia/expande países con utils.cleaning.expand_and_normalize_countries
# 2) Agrega por país y tipo
# 3) Ordena por Total y arma segmentos Top 1–10 / 11–20 / 21–30
# 4) Grafica barras horizontales agrupadas

# Salidas:
# - outputs/q3/q3_top01_10_grouped_barh.png
# - outputs/q3/q3_top11_20_grouped_barh.png
# - outputs/q3/q3_top21_30_grouped_barh.png


from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import plot_style as ps
from utils import cleaning as cl


# Agrupa por país y tipo, calcula totales y ordena por Total desc
def _pivot_country_type(df_expanded: pd.DataFrame) -> pd.DataFrame:
    grp = df_expanded.groupby(["country_final", "type"], as_index=False).size()
    pivot = grp.pivot(index="country_final", columns="type", values="size").fillna(0).astype(int)
    for col in ("Movie", "TV Show"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["Total"] = pivot["Movie"] + pivot["TV Show"]
    pivot = pivot.sort_values("Total", ascending=False)
    return pivot[["Movie", "TV Show", "Total"]]

# Devuelve tres cortes ordenados asc por Total para graficar 
def _slice_ranks(pivot_total: pd.DataFrame):
    top_1_10  = pivot_total.iloc[:10].copy().sort_values("Total", ascending=True)
    top_11_20 = pivot_total.iloc[10:20].copy().sort_values("Total", ascending=True)
    top_21_30 = pivot_total.iloc[20:30].copy().sort_values("Total", ascending=True)
    return top_1_10, top_11_20, top_21_30



def _add_value_labels(ax, y_positions, left_values, right_values, bar_h):
    for y, v in zip(y_positions, left_values):
        ax.text(v, y - bar_h/2, f"{int(v)}", va="center", ha="left",
                fontsize=9, color=ps.COLOR_TV)
    for y, v in zip(y_positions, right_values):
        ax.text(v, y + bar_h/2, f"{int(v)}", va="center", ha="left",
                fontsize=9, color=ps.COLOR_TV)


def _plot_grouped_barh(df_slice: pd.DataFrame, title: str, outpath: str) -> None:
    if df_slice.empty:
        return

    h = max(5, 0.48 * len(df_slice))  # alto dinámico
    plt.figure(figsize=(12, h), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    countries = df_slice.index.tolist()
    y = list(range(len(countries)))
    bar_h = 0.38

    ax.barh([i - bar_h/2 for i in y], df_slice["Movie"],   height=bar_h,
            label="Movies",   color=ps.COLOR_MOVIE)
    ax.barh([i + bar_h/2 for i in y], df_slice["TV Show"], height=bar_h,
            label="TV Shows", color=ps.COLOR_TV)

    ax.set_yticks(y)
    ax.set_yticklabels(countries)
    ax.set_xlabel("Cantidad de títulos", color=ps.COLOR_TV)
    ax.set_title(title, fontsize=13, color=ps.COLOR_TV)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True, min_n_ticks=8))
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.tick_params(axis="both", labelsize=9)

    _add_value_labels(ax, y, df_slice["Movie"].values, df_slice["TV Show"].values, bar_h)

    ax.legend()
    ps.add_source_note("Fuente: Netflix dataset. Elaboración propia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()


def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:
    outdir_q3 = os.path.join(outdir, "q3")
    os.makedirs(outdir_q3, exist_ok=True)

    if "country" not in df.columns or "type" not in df.columns:
        raise ValueError("El DataFrame debe contener 'country' y 'type'.")
    df_expanded = cl.expand_and_normalize_countries(df)
    pivot_total = _pivot_country_type(df_expanded)
    top_1_10, top_11_20, top_21_30 = _slice_ranks(pivot_total)

    _plot_grouped_barh(top_1_10,  "Top 1–10 países (Movies vs TV Shows)",
                       os.path.join(outdir_q3, "q3_top01_10_grouped_barh.png"))
    _plot_grouped_barh(top_11_20, "Top 11–20 países (Movies vs TV Shows)",
                       os.path.join(outdir_q3, "q3_top11_20_grouped_barh.png"))
    _plot_grouped_barh(top_21_30, "Top 21–30 países (Movies vs TV Shows)",
                       os.path.join(outdir_q3, "q3_top21_30_grouped_barh.png"))

    return {
        "pivot_total": pivot_total,
        "top1_10": top_1_10,
        "top11_20": top_11_20,
        "top21_30": top_21_30,
    }
