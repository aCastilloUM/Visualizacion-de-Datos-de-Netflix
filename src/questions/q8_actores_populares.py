# Pregunta 8:
# ¿Cuáles son los actores más populares?

# Pipeline:
# 1. Limpiar y expandir la columna 'cast' con expand_and_normalize_cast
# 2. Normalizar ratings con normalize_and_explode_ratings
# 3. Calcular ranking de actores por cantidad de títulos
# 4. Calcular pivotes por rating y proporciones
# 5. Graficar ranking, distribución por rating y heatmap

# Outputs:
# - outputs/q8/q8_top_actores_count_barh.png
# - outputs/q8/q8_top_actores_rating_stacked100.png
# - outputs/q8/q8_top_actores_rating_heatmap.png
# - outputs/q8/q8_top_actores_rating_donut.png
# - outputs/q8/q8_top_actores_type_donut.png

# Cleaning:
# - cl.expand_and_normalize_cast(df): Limpia y expande la columna 'cast' para agrupar correctamente los actores.
# - cl.normalize_and_explode_ratings(df): Normaliza los ratings y explota múltiples valores para mapear correctamente por actor.


from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import plot_style as ps
from utils import cleaning as cl

# Orden de ratings más comunes 
RATING_ORDER = [ "TV-MA","TV-14","TV-PG","PG-13","PG","R","G","TV-Y7","TV-Y","NR"]

# Devuelve DF con 'cast_final', 'type' y 'rating' listos 
def _prepare_cast_base(df: pd.DataFrame) -> pd.DataFrame:

    if "cast" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'cast'.")
    if "type" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'type'.")
    if "rating" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'rating'.")

    dfx = cl.expand_and_normalize_cast(df)
    dfx["type"] = dfx["type"].astype(str).str.strip()
    return dfx

# Índice con los nombres del Top-N actores por cantidad total de títulos.
def _top_actors(dfx: pd.DataFrame, topn: int = 20) -> pd.Index:
    counts = dfx.groupby("cast_final").size().sort_values(ascending=False)
    return counts.head(topn).index

# Conteo total por actor (solo Top-N)
def _pivot_actor_counts(dfx: pd.DataFrame, top_idx: pd.Index) -> pd.DataFrame:
    sub = dfx[dfx["cast_final"].isin(top_idx)]
    pv = sub.groupby("cast_final").size().sort_values(ascending=True)  # asc para barh
    return pv.to_frame(name="Total")

# Conteo por actor × rating_norm (solo Top-N).
def _pivot_actor_by_rating(dfx: pd.DataFrame, top_idx: pd.Index) -> pd.DataFrame:
    r = cl.normalize_and_explode_ratings(dfx)
    sub = r[r["cast_final"].isin(top_idx)]
    grp = sub.groupby(["cast_final", "rating_norm"], as_index=False).size()
    pv = grp.pivot(index="cast_final", columns="rating_norm", values="size").fillna(0).astype(int)

    # Reordenar columnas por RATING_ORDER y dejar al final cualquier rating raro
    cols_present = [c for c in RATING_ORDER if c in pv.columns]
    other_cols = [c for c in pv.columns if c not in cols_present]
    pv = pv[cols_present + other_cols]

    pv["Total"] = pv.sum(axis=1)
    pv = pv.sort_values("Total", ascending=True)  
    return pv

def _pivot_props_from_rating(pv_rating: pd.DataFrame) -> pd.DataFrame:

    df = pv_rating.drop(columns=["Total"]).copy()
    row_sum = df.sum(axis=1)
    props = df.div(row_sum, axis=0).fillna(0.0)
    props["Total"] = row_sum
    return props

def _plot_barh_top_counts(pv_counts: pd.DataFrame, outpath: str):
    if pv_counts.empty:
        return

    plt.figure(figsize=(12, max(5, 0.5 * len(pv_counts))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    ax.barh(pv_counts.index, pv_counts["Total"].values, color=ps.COLOR_MOVIE, edgecolor=ps.COLOR_TV, linewidth=0.6)

    ax.set_xlabel("Cantidad de títulos", color=ps.COLOR_TV)
    ax.set_title("Top actores por cantidad de títulos", fontsize=13, color=ps.COLOR_TV)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=6))

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG)
    plt.close()

def _plot_heatmap_actors_ratings(pv_rating: pd.DataFrame, outpath: str):
    if pv_rating.empty:
        return

    df = pv_rating.drop(columns=["Total"]).copy()
    ordered = [c for c in RATING_ORDER if c in df.columns] + [c for c in df.columns if c not in RATING_ORDER]
    df = df[ordered]

    plt.figure(figsize=(14, max(6, 0.55 * len(df))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    im = ax.imshow(df.values, aspect="auto")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist())
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns.tolist(), rotation=30, ha="right")

    ax.set_title("Heatmap: actores × rating (conteos)", fontsize=13, color=ps.COLOR_TV)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("# títulos")

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG, bbox_inches="tight")
    plt.close()

def _plot_donut(series: pd.Series, title: str, outpath: str):
    if series is None or series.empty:
        return

    s = series[series > 0].sort_values(ascending=False)
    if s.empty:
        return

    # Si hay demasiadas categorías, mostramos top-8 y agrupamos en Otros
    if len(s) > 8:
        top = s.iloc[:8]
        otros = pd.Series({"Otros": s.iloc[8:].sum()})
        s = pd.concat([top, otros])

    plt.figure(figsize=(8, 6), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    wedges, texts = ax.pie(
        s.values,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor=ps.COLOR_BG),
        labels=None  # no saturar con etiquetas
    )

    centre_circle = plt.Circle((0, 0), 0.62, fc=ps.COLOR_BG)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax.set_title(title, fontsize=13, color=ps.COLOR_TV)

    ax.legend(
        wedges, s.index.tolist(),
        title="Categorías",
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False
    )

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG, bbox_inches="tight")
    plt.close()


def _plot_donut_ratings(pv_rating: pd.DataFrame, outpath: str):
    if pv_rating is None or pv_rating.empty:
        return
    s = pv_rating.drop(columns=[c for c in ["Total"] if c in pv_rating.columns], errors="ignore").sum(axis=0)
    _plot_donut(s, "Distribución de ratings (actores Top)", outpath)


def _plot_donut_types(base: pd.DataFrame, top_idx: pd.Index, outpath: str):
    if base is None or base.empty or top_idx is None or len(top_idx) == 0:
        return
    sub = base[base["cast_final"].isin(top_idx)]
    s = sub["type"].value_counts()
    _plot_donut(s, "Distribución por tipo (actores Top)", outpath)


def run(df: pd.DataFrame, outdir: str = "outputs", topn: int = 20) -> dict:

    outdir_q8 = os.path.join(outdir, "q8")
    os.makedirs(outdir_q8, exist_ok=True)

    base = _prepare_cast_base(df)
    top_idx = _top_actors(base, topn=topn)

    pv_counts = _pivot_actor_counts(base, top_idx)

    pv_rating = _pivot_actor_by_rating(base, top_idx)
    props_rating = _pivot_props_from_rating(pv_rating)

    _plot_barh_top_counts(pv_counts, os.path.join(outdir_q8, "q8_top_actores_count_barh.png"))
    _plot_heatmap_actors_ratings(pv_rating, os.path.join(outdir_q8, "q8_top_actores_rating_heatmap.png"))
    _plot_donut_ratings( pv_rating, os.path.join(outdir_q8, "q8_top_actores_rating_donut.png"))
    _plot_donut_types(base, top_idx, os.path.join(outdir_q8, "q8_top_actores_type_donut.png"))

    return {
        "base": base,              
        "ranking": pv_counts,
        "pv_rating": pv_rating,
        "props_rating": props_rating,
    }