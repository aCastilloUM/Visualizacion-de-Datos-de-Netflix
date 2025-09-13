# -*- coding: utf-8 -*-
import os
import pandas as pd

# Importá cada pregunta como módulos independientes
from questions import q2_evolucion_estrenos as q2
from questions import q3_paises as q3

DATA_PATH = os.getenv(
    "DATA_PATH",
    r"C:\\Users\\agust\\OneDrive\\Escritorio\\Estudio\\Semestres\\6to Semestre\\Análisis de Datos\\netflix.csv"
)
OUTDIR = os.getenv("OUTDIR", "outputs")

def main():
    # Leer CSV una sola vez y pasar df a cada pregunta
    df = pd.read_csv(DATA_PATH)

    # ---- Pregunta 2 ----
    pivot_q2 = q2.run(df, outdir=OUTDIR)
    print("[Q2] Estrenos por año y tipo (primeras filas):")
    print(pivot_q2.head())
    print()
    # ---- Pregunta 3 ----
    pivot_q3 = q3.run(df, outdir=OUTDIR)
    print("[Q3] Top países (primeras filas):")
    print("[Q3] Resultados disponibles:", list(pivot_q3.keys()))
    # Mostramos un preview de cada uno
    for name, df_top in pivot_q3.items():
        print(f"\n[Q3] {name} (últimas filas para ver los más altos):")
        print(df_top.tail())  # df_top es un DataFrame
        print()
    # ---- Pregunta 4 ----
    # ---- Pregunta 5 ----
    # ---- Pregunta 6 ----
    # ---- Pregunta 7 ----
    # ---- Pregunta 8 ----
    # ---- Pregunta 9 ----
    # ---- Pregunta 10 ----


    # Aquí seguirías con:
    # from questions import q1_proporcion, q3_paises, ...
    # q1.run(df, outdir=OUTDIR)
    # q3.run(df, outdir=OUTDIR)
    # ...

if __name__ == "__main__":
    main()
