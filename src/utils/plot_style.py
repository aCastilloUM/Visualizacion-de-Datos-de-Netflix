# Utilidades de estilo para los gráficos (paleta Netflix y notas de fuente)

import matplotlib.pyplot as plt

# Paleta Netflix (centralizada aquí por si querés reusar en todo el proyecto)
COLOR_BG = "#f5f5f1"
COLOR_MOVIE = "#e50914"
COLOR_MOVIE_ALT = "#b20710"
COLOR_TV = "#221f1f"

def apply_netflix_style(ax=None):
    """
    Aplica estilo de fondo y grilla a un gráfico de Matplotlib.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_facecolor(COLOR_BG)
    plt.gcf().set_facecolor(COLOR_BG)
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    return ax

def add_source_note(text: str = "Fuente: Netflix dataset. Elaboración propia"):
    """
    Agrega una nota de fuente abajo a la derecha de la figura activa.
    """
    plt.figtext(
        0.99, 0.01, text,
        ha="right", va="bottom",
        fontsize=8, color=COLOR_TV, alpha=0.7
    )
