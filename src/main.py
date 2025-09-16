# -*- coding: utf-8 -*-
import os
import pandas as pd

# Importá cada pregunta como módulos independientes
from questions import q1_proporcion_peliculas_series as q1
from questions import q2_evolucion_estrenos as q2
from questions import q3_paises as q3
from questions import q4_rating_tipo as q4
from questions import q5_audiencias_paises as q5
from questions import q6_generos_estacionales as q6
from questions import q7_directores as q7
from questions import q8_actores_populares as q8
from questions import q9_duracion_contenido as q9
from questions import q10_palabras as q10

DATA_PATH = os.getenv(
    "DATA_PATH",
    r"C:\\Users\\agust\\OneDrive\\Escritorio\\Estudio\\Semestres\\6to Semestre\\Análisis de Datos\\netflix.csv"
)
OUTDIR = os.getenv("OUTDIR", "outputs")


def main():
    df = pd.read_csv(DATA_PATH)

    # --- Pregunta 1 ----
    pivot_q1 = q1.run(df, outdir=OUTDIR)
    print("[Q1] Proporciones (primeras filas):")
    print(pivot_q1.head())
    print()

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
        print(df_top.tail())  
        print()

    # ---- Pregunta 4 ----
    pivot_q4 = q4.run(df, outdir=OUTDIR)
    print("[Q4] Rating vs Tipo:")
    print(pivot_q4["counts"].head())  
    print()

    # ---- Pregunta 5 ----
    pivot_q5 = q5.run(df, outdir=OUTDIR)
    print("[Q5] Audiencia vs Pais:")
    print(pivot_q5["top1_10"].head())
    print()

    # ---- Pregunta 6 ----
    pivot_q6 = q6.run(df, outdir=OUTDIR)
    print("[Q6] tabla mes×categoría (shape):", pivot_q6["tabla_mes_categoria"].shape)
    print("[Q6] totales por mes (primeros):")
    print(pivot_q6["totales_mes"].head())
    print()

    # ---- Pregunta 7 ----
    pivot_q7 = q7.run(df, outdir=OUTDIR, topn=20)
    print("[Q7] Top 20 directores por tipo (primeras filas):")
    print(pivot_q7["pivot_tipo"].tail())  
    print()

    # ---- Pregunta 8 ----
    pivot_q8 = q8.run(df, outdir=OUTDIR)
    print("[Q8] Top actores por cantidad de títulos (primeras filas):")
    print(pivot_q8["ranking"].head())
    print()

    # ---- Pregunta 9 ----
    res_q9 = q9.run(df, outdir=OUTDIR)
    print("[Q9] Duración de películas y series:")
    print(res_q9["movies"].head())
    print(res_q9["tvshows"].head())
    print()


    # ---- Pregunta 10 ----
    res_q10 = q10.run(df, outdir=OUTDIR, topn=20)
    print("[Q10] Top palabras en títulos y descripciones:")
    print()
    print("Palabras en Títulos:")
    print()
    print(res_q10["top_words_titles"].tail())        # últimas = más frecuentes
    print()
    print("Palabras en Descripciones:")
    print()
    print(res_q10["top_words_descriptions"].tail())
    print()


if __name__ == "__main__":
    main()
