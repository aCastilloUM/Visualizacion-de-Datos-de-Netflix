# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

PALETA = {
    "fondo": "#f5f5f1",
    "grilla": "#221f1f",
    "acento": "#e50914",
}

# ---------------------------
# Preparación de datos
# ---------------------------
def preparar_estacionalidad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["date_added", "listed_in"])
    df["mes"] = df["date_added"].dt.month

    # Explode de categorías
    df["listed_in"] = df["listed_in"].astype(str).str.split(",")
    df = df.explode("listed_in")
    df["listed_in"] = df["listed_in"].str.strip()
    df = df[df["listed_in"].ne("")]

    return df

def _month_labels():
    return ["Ene","Feb","Mar","Abr","May","Jun",
            "Jul","Ago","Sep","Oct","Nov","Dic"]

# ---------------------------
# Gráficos
# ---------------------------
def heatmap_estacionalidad(df: pd.DataFrame, out_png_path: str):
    tabla = df.pivot_table(
        index="mes", columns="listed_in", values="title",
        aggfunc="count", fill_value=0
    ).reindex(range(1, 13), fill_value=0)

    plt.figure(figsize=(max(10, len(tabla.columns) * 0.35), 6), facecolor=PALETA["fondo"])
    ax = plt.gca()
    im = ax.imshow(tabla.values, aspect="auto")

    ax.set_yticks(range(12))
    ax.set_yticklabels(_month_labels())
    ax.set_xticks(range(len(tabla.columns)))
    ax.set_xticklabels(list(tabla.columns), rotation=30, ha="right")

    ax.set_title("Estacionalidad de estrenos por categoría (meses × categorías)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("# estrenos")

    plt.figtext(0.01, -0.02, "Fuente: Netflix dataset. Elaboración propia.",
                ha="left", va="top", fontsize=9)

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heatmap guardado en: {out_png_path}")

def barras_totales_por_mes(df: pd.DataFrame, out_png_path: str):
    totales_mes = df.groupby("mes")["title"].count().reindex(range(1, 13), fill_value=0)

    plt.figure(figsize=(9, 5), facecolor=PALETA["fondo"])
    ax = plt.gca()
    ax.bar(range(12), totales_mes.values, width=0.7, color=PALETA["acento"], edgecolor="black")

    ax.set_xticks(range(12))
    ax.set_xticklabels(_month_labels(), rotation=0)
    ax.set_ylabel("# estrenos")
    ax.set_title("Cantidad total de estrenos por mes (todas las categorías)")
    ax.grid(axis="y", color=PALETA["grilla"], alpha=0.15)

    plt.figtext(0.01, -0.02, "Fuente: Netflix dataset. Elaboración propia.",
                ha="left", va="top", fontsize=9)

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Barras por mes guardadas en: {out_png_path}")

# ---------------------------
# Función principal
# ---------------------------
def estacionalidad_generos(df: pd.DataFrame, outdir: str = "outputs"):
    base = preparar_estacionalidad(df)

    heatmap_estacionalidad(base, out_png_path=os.path.join(outdir, "q6_heatmap_categorias.png"))
    barras_totales_por_mes(base, out_png_path=os.path.join(outdir, "q6_barras_meses.png"))

    return None
