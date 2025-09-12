import pandas as pd
from ydata_profiling import ProfileReport
import os
import sys

csv_path = "C:\\Users\\agust\\OneDrive\\Escritorio\\Estudio\\Semestres\\6to Semestre\\Análisis de Datos\\Visualización_De_Datos_De_Netflix\\netflix.csv"
if not os.path.isfile(csv_path):
	print(f"ERROR: No se encontró el archivo '{csv_path}' en la carpeta actual.")
	sys.exit(1)

# 1. Cargar CSV
df = pd.read_csv(csv_path, low_memory=False)

# 2. Generar informe
profile = ProfileReport(df, title="Informe de Data Profiling - ", explorative=True)

# 3. Exportar a HTML
profile.to_file("informe_profiling.html")
