import os
import pandas as pd
import matplotlib.pyplot as plt

PALETA = {
    "fondo": "#f5f5f1",
    "acento": "#e50914",   # Películas
    "secundario": "#005f73",  # Series
    "grilla": "#221f1f",
}

def calcular_proporcion(df: pd.DataFrame) -> pd.DataFrame:
    # Normalización defensiva
    df = df.copy()
    df["type"] = df["type"].str.strip()
    df = df.dropna(subset=["release_year", "type"])
    df = df[df["release_year"].astype(str).str.isnumeric()]
    df["release_year"] = df["release_year"].astype(int)

    conteo = df.groupby(["release_year", "type"]).size().unstack(fill_value=0)
    conteo = conteo.sort_index()
    conteo["total"] = conteo.sum(axis=1)
    conteo["prop_peliculas"] = conteo.get("Movie", 0) / conteo["total"]
    conteo["prop_series"] = conteo.get("TV Show", 0) / conteo["total"]
    return conteo

def mostrar_tabla(conteo: pd.DataFrame, max_rows=10):
    """Muestra primeras filas de la tabla resumen en consola"""
    print("\nResumen proporción películas vs. series:\n")
    print(conteo[["Movie", "TV Show", "prop_peliculas", "prop_series"]].head(max_rows))
    print("...\n(Ver gráfico para evolución completa)")

def graficar_proporcion(conteo: pd.DataFrame, outdir="outputs"):
    plt.figure(figsize=(10, 6), facecolor=PALETA["fondo"])
    ax = plt.gca()
    ax.plot(conteo.index, conteo["prop_peliculas"], label="Películas",
            linewidth=2, color=PALETA["acento"])
    ax.plot(conteo.index, conteo["prop_series"], label="Series",
            linewidth=2, color=PALETA["secundario"])

    ax.set_ylim(0, 1)
    ax.set_yticks([i/10 for i in range(0, 11)])
    ax.set_yticklabels([f"{int(v*100)}%" for v in ax.get_yticks()])

    ax.set_xlabel("Año de lanzamiento")
    ax.set_ylabel("Proporción")
    ax.set_title("Proporción de películas y series por año")
    ax.grid(True, color=PALETA["grilla"], alpha=0.15)
    ax.legend(title="Tipo de contenido")

    # Subtítulo / fuente (texto pequeño bajo el gráfico)
    plt.figtext(0.01, -0.02,
        "Fuente: Netflix dataset.",
        ha="left", va="top", fontsize=9)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "q1_proporcion_peliculas_series.png"), dpi=200, bbox_inches="tight")
    plt.close()
