# -*- coding: utf-8 -*-
import os
import pandas as pd

# Importá cada pregunta como módulos independientes
from questions import q2_evolucion_estrenos as q2

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

    # Aquí seguirías con:
    # from questions import q1_proporcion, q3_paises, ...
    # q1.run(df, outdir=OUTDIR)
    # q3.run(df, outdir=OUTDIR)
    # ...

if __name__ == "__main__":
    main()
