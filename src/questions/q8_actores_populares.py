# -*- coding: utf-8 -*-
# Pregunta 8:
# ¿Cuáles son los actores más populares?
# - Top-N por cantidad de títulos (conteo bruto)
# - Distribución por rating para esos actores (barras apiladas 100%)
# - Heatmap actores × rating (conteos)
#
# Limpieza reutilizable:
# - utils.cleaning.expand_and_normalize_cast  (cast -> cast_final)
# - utils.cleaning.normalize_and_explode_ratings  (rating -> rating_norm)
#
# Estilo:
# - utils.plot_style (paleta Netflix + add_source_note)
#
# Salidas:
# - outputs/q8/q8_top_actores_count_barh.png             (ranking por conteo)
# - outputs/q8/q8_top_actores_rating_stacked100.png      (distribución por rating, 100%)
# - outputs/q8/q8_top_actores_rating_heatmap.png         (heatmap actores × rating)
#
# Interfaz:
# - run(df, outdir="outputs", topn=20) -> dict con pivots/tablas para revisar
#     Pipeline Q8:
#    1) Expande/normaliza el elenco (cast_final).
#    2) Selecciona Top-N actores por total de títulos.
#    3) Genera:
#       - Ranking por conteo (barh)
#       - Distribución por rating 100% (stacked barh)
#       - Heatmap actores × rating (conteos)
#    Devuelve dict con DataFrames útiles.

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

def _top_actors(dfx: pd.DataFrame, topn: int = 20) -> pd.Index:
    counts = dfx.groupby("cast_final").size().sort_values(ascending=False)
    return counts.head(topn).index

# Pivots
def _pivot_actor_counts(dfx: pd.DataFrame, top_idx: pd.Index) -> pd.DataFrame:
    """
    Conteo total de títulos por actor (Top-N).
    """
    sub = dfx[dfx["cast_final"].isin(top_idx)]
    pv = sub.groupby("cast_final").size().sort_values(ascending=True)  # asc para barh
    return pv.to_frame(name="Total")

def _pivot_actor_by_rating(dfx: pd.DataFrame, top_idx: pd.Index) -> pd.DataFrame:
    """
    Conteo por actor × rating_norm (solo Top-N). 
    Usa ratings normalizados y explota múltiples si los hubiera.
    """
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

# Plot
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

def _plot_stacked100_by_rating(props_rating: pd.DataFrame, outpath: str):
    """
    Barras horizontales apiladas 100%: distribución por rating para Top actores.
    """
    if props_rating.empty:
        return

    df = props_rating.copy()
    # Ordenar columnas: ratings primero (según RATING_ORDER), ignorando 'Total'
    cols = [c for c in df.columns if c != "Total"]
    plt.figure(figsize=(14, max(6, 0.55 * len(df))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    y = list(range(len(df)))
    bar_h = 0.6
    left = [0.0] * len(df)

    # Apilar cada rating como segmento
    for col in cols:
        vals = df[col].values
        ax.barh(y, vals, height=bar_h, left=left, label=col)
        left = [l + v for l, v in zip(left, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels(df.index.tolist())
    ax.set_xlabel("Proporción por rating", color=ps.COLOR_TV)
    ax.set_title("Distribución por rating (100%) para actores Top", fontsize=13, color=ps.COLOR_TV)

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([f"{int(v*100)}%" for v in [0, 25, 50, 75, 100]])

    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, facecolor=ps.COLOR_BG, bbox_inches="tight")
    plt.close()

def _plot_heatmap_actors_ratings(pv_rating: pd.DataFrame, outpath: str):
    if pv_rating.empty:
        return

    df = pv_rating.drop(columns=["Total"]).copy()
    # Asegurar orden columnas por RATING_ORDER + otros
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
    """
    Donut chart genérico para una Serie (index=categorías, values=conteos).
    Oculta etiquetas si hay muchas categorías y deja leyenda a la derecha.
    """
    if series is None or series.empty:
        return

    # Ordenar desc y filtrar ceros
    s = series[series > 0].sort_values(ascending=False)
    if s.empty:
        return

    # Si hay demasiadas categorías, mostramos top-8 y agrupamos 'Otros'
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
        labels=None  # evitamos saturar con etiquetas
    )

    # Donut: círculo central
    centre_circle = plt.Circle((0, 0), 0.62, fc=ps.COLOR_BG)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax.set_title(title, fontsize=13, color=ps.COLOR_TV)

    # Leyenda a la derecha
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
    """
    Donut de distribución de ratings entre todos los títulos de los actores TOP.
    Usa la tabla actor × rating (pv_rating) ya construida.
    """
    if pv_rating is None or pv_rating.empty:
        return
    # Sumamos por columna (rating). Quitamos 'Total' si está.
    s = pv_rating.drop(columns=[c for c in ["Total"] if c in pv_rating.columns], errors="ignore").sum(axis=0)
    _plot_donut(s, "Distribución de ratings (actores Top)", outpath)


def _plot_donut_types(base: pd.DataFrame, top_idx: pd.Index, outpath: str):
    """
    Donut de distribución por tipo (Movie vs TV Show) SOLO para títulos de actores Top.
    """
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

    # Ranking (conteo)
    pv_counts = _pivot_actor_counts(base, top_idx)

    # Distribución por rating (conteos y 100%)
    pv_rating = _pivot_actor_by_rating(base, top_idx)
    props_rating = _pivot_props_from_rating(pv_rating)

    # Gráficos
    _plot_barh_top_counts(pv_counts, os.path.join(outdir_q8, "q8_top_actores_count_barh.png"))
    _plot_stacked100_by_rating(props_rating, os.path.join(outdir_q8, "q8_top_actores_rating_stacked100.png"))
    _plot_heatmap_actors_ratings(pv_rating, os.path.join(outdir_q8, "q8_top_actores_rating_heatmap.png"))
    # Donuts (círculos) de resumen
    _plot_donut_ratings( pv_rating, os.path.join(outdir_q8, "q8_top_actores_rating_donut.png"))
    _plot_donut_types(base, top_idx, os.path.join(outdir_q8, "q8_top_actores_type_donut.png"))

    return {
        "base": base,              # DF expandido con cast_final
        "ranking": pv_counts,      # actor × Total (asc para barh)
        "pv_rating": pv_rating,    # actor × ratings (+ Total)
        "props_rating": props_rating,  # proporciones por rating (filas suman 1)
    }