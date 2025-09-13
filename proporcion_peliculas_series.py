import pandas as pd
import matplotlib.pyplot as plt

# Asume que el archivo ya fue leído y el DataFrame se llama df
# Si no, descomenta la siguiente línea y ajusta la ruta:
# df = pd.read_csv('c:/Users/Usuario/Downloads/netflix (1).csv')

# 2. Calcular la proporción de películas y series por año
def calcular_proporcion(df):
    conteo = df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
    conteo['total'] = conteo.sum(axis=1)
    conteo['proporcion_peliculas'] = conteo.get('Movie', 0) / conteo['total']
    conteo['proporcion_series'] = conteo.get('TV Show', 0) / conteo['total']
    return conteo

# 3. Mostrar tabla resumen
def mostrar_tabla(conteo):
    print(conteo[['Movie', 'TV Show', 'proporcion_peliculas', 'proporcion_series']])

# 4. Graficar la evolución de la proporción
def graficar_proporcion(conteo):
    # Paleta de colores recomendada
    colores = ['#f5f5f1', '#e50914', '#b20710', '#221f1f']
    plt.figure(figsize=(10,6))
    plt.plot(conteo.index, conteo['proporcion_peliculas'], label='Películas', marker='o', color=colores[1])
    plt.plot(conteo.index, conteo['proporcion_series'], label='Series', marker='o', color=colores[2])
    plt.xlabel('Año de lanzamiento')
    plt.ylabel('Proporción')
    plt.title('Proporción de Películas y Series por Año')
    plt.legend()
    plt.grid(True, color=colores[3], alpha=0.2)
    plt.gca().set_facecolor(colores[0])
    plt.tight_layout()
    plt.show()

# Ejemplo de uso:
# conteo = calcular_proporcion(df)
# mostrar_tabla(conteo)
# graficar_proporcion(conteo)
