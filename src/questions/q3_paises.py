# -*- coding: utf-8 -*-
# Pregunta 3 (sin auditorías ni CSVs):
# - Solo genera gráficos (PNG) en outputs/q3/
# - Expande coproducciones: "A, B" -> A y B
# - Normaliza alias de países
# - Segmentos: Top 1–10, Top 11–20, Top 21–30
# - Paleta Netflix + nota de fuente

from __future__ import annotations
import os
import re
import unicodedata
import difflib
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import plot_style as ps  # <<< estilos centralizados

# Alias de países (correcciones comunes)
COUNTRY_ALIASES: Dict[str, str] = {
    "Usa": "United States",
    "U.s.": "United States",
    "U.s.a.": "United States",
    "United States Of America": "United States",
    "England": "United Kingdom",
    "Uk": "United Kingdom",
    "U.k.": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Britain": "United Kingdom",
    "Korea, South": "South Korea",
    "Republic Of Korea": "South Korea",
    "Korea": "South Korea",
    "Viet Nam": "Vietnam",
    "Russian Federation": "Russia",
    "Uae": "United Arab Emirates",
    "Iran, Islamic Republic Of": "Iran",
    "Hong Kong S.a.r.": "Hong Kong",
    "Taiwan, Province Of China": "Taiwan",
    "Macau, Sar China": "Macau",
    "Czechia": "Czech Republic",
    "Moldova, Republic Of": "Moldova",
    "Tanzania, United Republic Of": "Tanzania",
    "Bolivia, Plurinational State Of": "Bolivia",
    "Venezuela, Bolivarian Republic Of": "Venezuela",
    "Syrian Arab Republic": "Syria",
    # Español
    "Estados Unidos": "United States",
    "Reino Unido": "United Kingdom",
    "Corea Del Sur": "South Korea",
    "Emiratos Arabes Unidos": "United Arab Emirates",
    "Republica Checa": "Czech Republic",
}

# Lista canónica (para sugerencias)
CANON_COUNTRIES: List[str] = [
    "United States","United Kingdom","India","Japan","South Korea","Canada","France","Spain","Germany",
    "Italy","Mexico","Turkey","Brazil","Australia","Argentina","Colombia","Chile","Peru","Russia",
    "China","Hong Kong","Taiwan","Thailand","Philippines","Indonesia","Malaysia","Singapore","Vietnam",
    "United Arab Emirates","Saudi Arabia","Egypt","South Africa","Nigeria","Kenya","Ghana","Morocco",
    "Netherlands","Belgium","Sweden","Norway","Denmark","Finland","Poland","Czech Republic","Austria",
    "Switzerland","Portugal","Ireland","Greece","Israel","Iran","Iraq","Syria","Lebanon","Jordan",
    "New Zealand","Bangladesh","Pakistan","Sri Lanka","Nepal","Romania","Hungary","Bulgaria","Serbia",
    "Croatia","Slovakia","Slovenia","Ukraine","Belarus","Lithuania","Latvia","Estonia","Iceland",
    "Luxembourg","Uruguay","Paraguay","Bolivia","Ecuador","Venezuela","Guatemala","Costa Rica",
    "Panama","El Salvador","Honduras","Nicaragua","Dominican Republic","Cuba","Haiti","Jamaica",
    "Qatar","Kuwait","Bahrain","Oman","Yemen","Algeria","Tunisia","Libya","Ethiopia","Tanzania",
    "Uganda","Zimbabwe","Zambia","Botswana","Namibia","Cameroon","Ivory Coast","Senegal","Mali",
    "Armenia","Azerbaijan","Georgia","Kazakhstan","Uzbekistan","Kyrgyzstan","Mongolia",
    "Macau","Malta","Cyprus","Liechtenstein","Andorra","Monaco","San Marino","Vatican City"
]

# ----------------- Limpieza / Normalización -----------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _clean_country_token(tok: str) -> str:
    t = tok.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("'", "")
    t = _strip_accents(t).title()
    t = re.sub(r"\s*\(.*?\)\s*$", "", t).strip()
    return t

def _correct_alias(tok: str) -> str:
    return COUNTRY_ALIASES.get(tok, tok)

def _suggest_if_needed(tok: str) -> str:
    if tok in CANON_COUNTRIES:
        return tok
    cand = difflib.get_close_matches(tok, CANON_COUNTRIES, n=1, cutoff=0.85)
    return cand[0] if cand else tok

def _expand_countries(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["country"] = df2["country"].fillna("").astype(str)
    df2["country_tokens"] = df2["country"].str.split(",")
    df2 = df2.explode("country_tokens", ignore_index=True)
    df2["country_final"] = (
        df2["country_tokens"]
        .map(lambda x: _clean_country_token(x) if isinstance(x, str) else "")
        .map(_correct_alias)
        .map(_suggest_if_needed)
    )
    df2 = df2[df2["country_final"] != ""]
    return df2

# ----------------- Agregación / Slices -----------------

def _pivot_country_type(df_expanded: pd.DataFrame) -> pd.DataFrame:
    grp = df_expanded.groupby(["country_final", "type"], as_index=False).size()
    pivot = grp.pivot(index="country_final", columns="type", values="size").fillna(0).astype(int)
    for col in ("Movie", "TV Show"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["Total"] = pivot["Movie"] + pivot["TV Show"]
    pivot = pivot.sort_values("Total", ascending=False)
    return pivot

def _slice_ranks(pivot_total: pd.DataFrame):
    top_1_10  = pivot_total.iloc[:10].copy().sort_values("Total", ascending=True)
    top_11_20 = pivot_total.iloc[10:20].copy().sort_values("Total", ascending=True)
    top_21_30 = pivot_total.iloc[20:30].copy().sort_values("Total", ascending=True)
    return top_1_10, top_11_20, top_21_30

# ----------------- Plot -----------------

def _add_value_labels(ax, y_positions, left_values, right_values, bar_h):
    for y, v in zip(y_positions, left_values):
        ax.text(v, y - bar_h/2, f"{int(v)}", va="center", ha="left", fontsize=9, color=ps.COLOR_TV)
    for y, v in zip(y_positions, right_values):
        ax.text(v, y + bar_h/2, f"{int(v)}", va="center", ha="left", fontsize=9, color=ps.COLOR_TV)

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

    ax.barh([i - bar_h/2 for i in y], df_slice["Movie"],   height=bar_h, label="Movies",   color=ps.COLOR_MOVIE)
    ax.barh([i + bar_h/2 for i in y], df_slice["TV Show"], height=bar_h, label="TV Shows", color=ps.COLOR_TV)

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

# ----------------- API -----------------

def run(df: pd.DataFrame, outdir: str = "outputs") -> dict:
    outdir_q3 = os.path.join(outdir, "q3")
    os.makedirs(outdir_q3, exist_ok=True)

    if "country" not in df.columns or "type" not in df.columns:
        raise ValueError("El DataFrame debe contener 'country' y 'type'.")

    df_expanded = _expand_countries(df)
    pivot_total = _pivot_country_type(df_expanded)

    top_1_10, top_11_20, top_21_30 = _slice_ranks(pivot_total)

    _plot_grouped_barh(top_1_10,  "Top 1–10 países (Movies vs TV Shows)",  os.path.join(outdir_q3, "q3_top01_10_grouped_barh.png"))
    _plot_grouped_barh(top_11_20, "Top 11–20 países (Movies vs TV Shows)", os.path.join(outdir_q3, "q3_top11_20_grouped_barh.png"))
    _plot_grouped_barh(top_21_30, "Top 21–30 países (Movies vs TV Shows)", os.path.join(outdir_q3, "q3_top21_30_grouped_barh.png"))

    return {
        "top1_10": top_1_10,
        "top11_20": top_11_20,
        "top21_30": top_21_30,
    }


