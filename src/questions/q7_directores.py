# Pregunta 7:
# ¿Qué directores tienen más títulos? ¿Se concentran en algún tipo de contenido
# o en una audiencia específica?
#
# - Limpia/expande 'director' (utils.cleaning.expand_and_normalize_directors)
# - Para audiencia usa ratings normalizados (utils.cleaning.normalize_and_explode_ratings
#   + utils.cleaning.map_rating_to_audience(mode="adult_kids"))
# - Gráficos:
#     * outputs/q7/q7_top20_directores_tipo_stacked.png       (Movie vs TV Show)
#     * outputs/q7/q7_top20_directores_audiencia_stacked.png  (Adulto vs Infantil)
# - Interfaz: run(df, outdir="outputs") -> dict con pivots y ranking

#   Pipeline Q7:
#    1) Expande/normaliza directores
#    2) Selecciona Top-N por total de títulos
#    3) Genera:
#       - Stacked (Movie vs TV Show)
#       - Stacked (Adulto vs Infantil)
#    4) Devuelve dict con pivots y ranking

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps
from utils import cleaning as cl

# Preparación de datos
def _prepare_directors_base(df: pd.DataFrame) -> pd.DataFrame:

    if "director" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'director'.")
    if "type" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'type'.")
    if "rating" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'rating'.")

    dfx = cl.expand_and_normalize_directors(df)
    # Asegurar 'type' limpia
    dfx["type"] = dfx["type"].astype(str).str.strip()
    return dfx

#   Retorna el nombre de los Top-N directores por cantidad total de títulos.

def _top_directors(dfx: pd.DataFrame, topn: int = 20) -> pd.Index:

    counts = dfx.groupby("director_final").size().sort_values(ascending=False)
    return counts.head(topn).index

# Pivots

def _pivot_director_type(dfx: pd.DataFrame, top_index: pd.Index) -> pd.DataFrame:

    sub = dfx[dfx["director_final"].isin(top_index)]
    grp = sub.groupby(["director_final", "type"], as_index=False).size()
    pv = grp.pivot(index="director_final", columns="type", values="size").fillna(0).astype(int)
    for col in ("Movie", "TV Show"):
        if col not in pv.columns:
            pv[col] = 0
    pv["Total"] = pv["Movie"] + pv["TV Show"]
    # orden por total desc; luego asc para la visualización (barh)
    pv = pv.sort_values("Total", ascending=True)
    return pv[["Movie", "TV Show", "Total"]]

def _pivot_director_audience(dfx: pd.DataFrame, top_index: pd.Index) -> pd.DataFrame:

    # Normalizar y explotar ratings
    r = cl.normalize_and_explode_ratings(dfx)
    r["audiencia"] = r["rating_norm"].map(lambda x: cl.map_rating_to_audience(x, mode="adult_kids"))

    sub = r[r["director_final"].isin(top_index)]
    grp = sub.groupby(["director_final", "audiencia"], as_index=False).size()
    pv = grp.pivot(index="director_final", columns="audiencia", values="size").fillna(0).astype(int)

    # asegurar columnas
    for col in ("Adulto", "Infantil"):
        if col not in pv.columns:
            pv[col] = 0
    pv["Total"] = pv["Adulto"] + pv["Infantil"]
    pv = pv.sort_values("Total", ascending=True)
    return pv[["Infantil", "Adulto", "Total"]]

# Plot

def _plot_stacked_barh(df_slice: pd.DataFrame, left_col: str, right_col: str, title: str, outpath: str):

    if df_slice.empty:
        return

    plt.figure(figsize=(12, max(6, 0.55 * len(df_slice))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    directors = df_slice.index.tolist()
    y = list(range(len(directors)))
    bar_h = 0.6

    left_vals  = df_slice[left_col].values
    right_vals = df_slice[right_col].values

    ax.barh(y, left_vals,  height=bar_h, color=ps.COLOR_MOVIE, label=left_col)
    ax.barh(y, right_vals, height=bar_h, left=left_vals, color=ps.COLOR_TV, label=right_col)

    ax.set_yticks(y)
    ax.set_yticklabels(directors)
    ax.set_xlabel("Cantidad de títulos", color=ps.COLOR_TV)
    ax.set_title(title, fontsize=13, color=ps.COLOR_TV)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, min_n_ticks=6))
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.legend(loc="lower right")

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()


def run(df: pd.DataFrame, outdir: str = "outputs", topn: int = 20) -> dict:
    outdir_q7 = os.path.join(outdir, "q7")
    os.makedirs(outdir_q7, exist_ok=True)

    base = _prepare_directors_base(df)
    top_idx = _top_directors(base, topn=topn)

    # Pivots
    pv_tipo      = _pivot_director_type(base, top_idx)
    pv_audiencia = _pivot_director_audience(base, top_idx)

    # Gráficos
    _plot_stacked_barh(
        pv_tipo,
        left_col="Movie", right_col="TV Show",
        title=f"Top {topn} directores por tipo (Movie vs TV Show)",
        outpath=os.path.join(outdir_q7, "q7_top20_directores_tipo_stacked.png"),
    )
    _plot_stacked_barh(
        pv_audiencia,
        left_col="Infantil", right_col="Adulto",
        title=f"Top {topn} directores por audiencia (Infantil vs Adulto)",
        outpath=os.path.join(outdir_q7, "q7_top20_directores_audiencia_stacked.png"),
    )

    return {
        "base": base,                 # DF con director_final y columnas originales
        "ranking": top_idx,           # Índice con los nombres del Top-N
        "pivot_tipo": pv_tipo,        # director × (Movie, TV Show, Total)
        "pivot_audiencia": pv_audiencia,  # director × (Infantil, Adulto, Total)
    }
