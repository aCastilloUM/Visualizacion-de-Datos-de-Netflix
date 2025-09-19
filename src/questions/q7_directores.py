# Pregunta 7:
# ¿Qué directores tienen más títulos? ¿Se concentran en algún tipo de contenido o en una audiencia específica?

# Pipeline:
# 1. Limpiar y expandir la columna 'director' con expand_and_normalize_directors
# 2. Normalizar ratings y mapear a audiencias con normalize_and_explode_ratings y map_rating_to_audience
# 3. Explotar géneros y mapear a género canónico con explode_listed_in y add_genre_from_listed_in
# 4. Calcular ranking y pivotes por tipo, audiencia y género
# 5. Graficar los resultados

# Outputs:
# - outputs/q7/q7_top20_directores_tipo_stacked.png
# - outputs/q7/q7_top20_directores_audiencia_stacked.png
# - outputs/q7/q7_top20_directores_genero_dominante.png

# Cleaning:
# - cl.expand_and_normalize_directors(df): Limpia y expande la columna 'director' para agrupar correctamente.
# - cl.normalize_and_explode_ratings(df): Normaliza los ratings y explota múltiples valores para mapear audiencias.
# - cl.map_rating_to_audience(): Mapea los ratings normalizados a categorías de audiencia.
# - cl.explode_listed_in(df): Explota la columna de géneros para analizar cada género por separado.
# - cl.add_genre_from_listed_in(df): Mapea los géneros a una categoría canónica para análisis.

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import plot_style as ps
from utils import cleaning as cl


# Devuelve un DF expandido por director con 'director_final' y columnas originales.
def _prepare_directors_base(df: pd.DataFrame) -> pd.DataFrame:
    required = {"director", "type", "rating"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    dfx = cl.expand_and_normalize_directors(df)
    dfx["type"] = dfx["type"].astype(str).str.strip()
    return dfx

# Índice con los nombres del Top-N directores por cantidad total de títulos.
def _top_directors(dfx: pd.DataFrame, topn: int = 20) -> pd.Index:
    counts = dfx.groupby("director_final").size().sort_values(ascending=False)
    return counts.head(topn).index


# Obitne conteos por tipo de contenido (Movie/TV Show) para los directores en top_index
def _pivot_director_type(dfx: pd.DataFrame, top_index: pd.Index) -> pd.DataFrame:
    sub = dfx[dfx["director_final"].isin(top_index)]
    grp = sub.groupby(["director_final", "type"], as_index=False).size()
    pv = grp.pivot(index="director_final", columns="type", values="size").fillna(0).astype(int)

    for col in ("Movie", "TV Show"):
        if col not in pv.columns:
            pv[col] = 0

    pv["Total"] = pv["Movie"] + pv["TV Show"]
    # orden asc para barh (de menor a mayor)
    pv = pv.sort_values("Total", ascending=True)
    return pv[["Movie", "TV Show", "Total"]]

# Obtiene conteos por audiencia (Adulto/Infantil) para los directores en top_index
def _pivot_director_audience(dfx: pd.DataFrame, top_index: pd.Index) -> pd.DataFrame:
    r = cl.normalize_and_explode_ratings(dfx)
    r["audiencia"] = r["rating_norm"].map(lambda x: cl.map_rating_to_audience(x, mode="adult_kids"))

    sub = r[r["director_final"].isin(top_index)]
    grp = sub.groupby(["director_final", "audiencia"], as_index=False).size()
    pv = grp.pivot(index="director_final", columns="audiencia", values="size").fillna(0).astype(int)

    for col in ("Adulto", "Infantil"):
        if col not in pv.columns:
            pv[col] = 0

    pv["Total"] = pv["Adulto"] + pv["Infantil"]
    pv = pv.sort_values("Total", ascending=True)
    return pv[["Infantil", "Adulto", "Total"]]



# Colores para el gráfico de género dominante 
_GENRE_COLOR = {
    "Comedia": "#1FA76C",
    "Drama": "#374151",
    "Terror": "#111111",
    "Thriller": "#4b5563",
    "Crimen/Misterio": "#1f2937",
    "Acción/Aventura": "#ef4444",
    "Ciencia Ficción/Fantasía": "#10b981",
    "Romance": "#ec4899",
    "Documental": "#6b7280",
    "Ciencia/Naturaleza": "#059669",
    "Infantil/Familiar": "#3b82f6",
    "Anime": "#8b5cf6",
    "Música": "#22d3ee",
    "Deportes": "#14b8a6",
    "Reality": "#64748b",
    "Stand-Up": "#e50914",
    "Clásicos/Culto": "#9CA3AF",
    "Independiente": "#A3A3A3",
    "Fe/Espiritualidad": "#94A3B8",
    "Internacional/Regional": "#6B7280",
    "TV (General)": "#9CA3AF",
}

_MARKER_GENRES = {"Internacional/Regional", "TV (General)"}


def _pivot_director_genre(df_original: pd.DataFrame,
                          top_index: pd.Index,
                          drop_markers: bool = True) -> pd.DataFrame:
    dfx = cl.explode_listed_in(df_original)
    dfx = cl.add_genre_from_listed_in(dfx)
    dfx = cl.expand_and_normalize_directors(dfx)

    dfx = dfx[dfx["director_final"].isin(top_index)].copy()
    if dfx.empty:
        return pd.DataFrame(columns=["genre", "count"]).set_index(pd.Index([], name="director_final"))

    grp = (
        dfx.groupby(["director_final", "genre_main"])
           .size()
           .reset_index(name="count")
           .rename(columns={"genre_main": "genre"})
    )

    
    if drop_markers:
        pref = grp[~grp["genre"].isin(_MARKER_GENRES)].copy()
        missing = set(grp["director_final"]) - set(pref["director_final"])
        backfill = grp[grp["director_final"].isin(missing)]
        candidates = pd.concat([pref, backfill], ignore_index=True)
    else:
        candidates = grp

    candidates = candidates.sort_values(["director_final", "count", "genre"],
                                        ascending=[True, False, True])
    dominant = candidates.drop_duplicates(subset=["director_final"], keep="first")

    dominant = dominant.set_index("director_final").loc[top_index]
    return dominant[["genre", "count"]]




def _plot_stacked_barh(df_slice: pd.DataFrame, left_col: str, right_col: str, title: str, outpath: str):
    if df_slice.empty:
        return

    plt.figure(figsize=(12, max(6, 0.55 * len(df_slice))), facecolor=ps.COLOR_BG)
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    directors = df_slice.index.tolist()
    y = list(range(len(directors)))
    bar_h = 0.6

    left_vals = df_slice[left_col].values
    right_vals = df_slice[right_col].values

    ax.barh(y, left_vals, height=bar_h, color=ps.COLOR_MOVIE, label=left_col)
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


def _plot_director_dominant_genre(
    df_dom: pd.DataFrame,
    outpath: str,
    title: str = "Género dominante por director (Top 20)"
):
   
    if df_dom.empty:
        return

    df_dom = df_dom.sort_values("count", ascending=False)

    plt.figure(figsize=(12, max(6, 0.55 * len(df_dom))))
    ax = plt.gca()
    ps.apply_netflix_style(ax)

    directors = df_dom.index.tolist()
    y = list(range(len(directors)))
    counts = df_dom["count"].values
    genres = df_dom["genre"].tolist()

    colors = [_GENRE_COLOR.get(g, "#9ca3af") for g in genres]

    ax.barh(y, counts, height=0.6, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(directors)
    ax.set_xlabel("Cantidad de títulos en el género dominante")
    ax.set_title(title, fontsize=13)

    ax.invert_yaxis()

    for i, (val, g) in enumerate(zip(counts, genres)):
        if val and val > 0:
            ax.text(val + 0.1, i, g, va="center", fontsize=9)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, min_n_ticks=6))
    ax.grid(axis="x", alpha=0.35, linestyle="--")

    present = sorted(set(genres))
    handles = [plt.Rectangle((0, 0), 1, 1, color=_GENRE_COLOR.get(g, "#9ca3af")) for g in present]
    ax.legend(handles, present, title="Género", loc="lower right", fontsize=9)

    ps.add_source_note()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()



def run(df: pd.DataFrame, outdir: str = "outputs", topn: int = 20) -> dict:
    outdir_q7 = os.path.join(outdir, "q7")
    os.makedirs(outdir_q7, exist_ok=True)

    base = _prepare_directors_base(df)
    top_idx = _top_directors(base, topn=topn)

    pv_tipo = _pivot_director_type(base, top_idx)
    pv_audiencia = _pivot_director_audience(base, top_idx)

    dom_genre = _pivot_director_genre(df, top_idx, drop_markers=True)

    _plot_stacked_barh(
        pv_tipo,
        left_col="Movie",
        right_col="TV Show",
        title=f"Top {topn} directores por tipo (Movie vs TV Show)",
        outpath=os.path.join(outdir_q7, "q7_top20_directores_tipo_stacked.png"),
    )

    _plot_stacked_barh(
        pv_audiencia,
        left_col="Infantil",
        right_col="Adulto",
        title=f"Top {topn} directores por audiencia (Infantil vs Adulto)",
        outpath=os.path.join(outdir_q7, "q7_top20_directores_audiencia_stacked.png"),
    )

    _plot_director_dominant_genre(
        dom_genre,
        outpath=os.path.join(outdir_q7, "q7_top20_directores_genero_dominante.png"),
        title=f"Género dominante por director (Top {topn})",
    )

    return {
        "base": base,                 
        "ranking": top_idx,             
        "pivot_tipo": pv_tipo,         
        "pivot_audiencia": pv_audiencia,   
        "dominant_genre": dom_genre,  
    }
