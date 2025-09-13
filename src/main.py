# -*- coding: utf-8 -*-
import os
import pandas as pd

# Importá cada pregunta como módulos independientes
from questions import q1_proporcion_peliculas_series, q2_evolucion_estrenos, q6_generos_estacionales

DATA_PATH = os.getenv(
    "DATA_PATH",
    r"C:\Users\Usuario\OneDrive - Universidad de Montevideo\Analisis de Datos\netflix (2).csv"
)
OUTDIR = os.getenv("OUTDIR", "outputs")

def main():
    # Leer CSV una sola vez y pasar df a cada pregunta
    df = pd.read_csv(DATA_PATH)

    # ---- Pregunta 1 ----
    print("question 1")
    conteo = q1_proporcion_peliculas_series.calcular_proporcion(df)
    q1_proporcion_peliculas_series.mostrar_tabla(conteo)
    q1_proporcion_peliculas_series.graficar_proporcion(conteo, outdir=OUTDIR)

    print("\n")

    # ---- Pregunta 2 ----
    print("question 2")
    pivot_q2 = q2_evolucion_estrenos.run(df, outdir=OUTDIR)
    print("[Q2] Estrenos por año y tipo (primeras filas):")
    print(pivot_q2.head())

    print("\n")

    # ---- Pregunta 6 ----
    print("question 6")
    pivot_q6 = q6_generos_estacionales.estacionalidad_generos(df, outdir=OUTDIR)

    # Aquí seguirías con:
    # from questions import q1_proporcion, q3_paises, ...
    # q1.run(df, outdir=OUTDIR)
    # q3.run(df, outdir=OUTDIR)
    # ...

if __name__ == "__main__":
    main()
