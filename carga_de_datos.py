# carga_datos.py
import pandas as pd
from pathlib import Path
import re

def cargar_datos():
    ruta = Path("EPH")
    lista_df = []
    
    # 1. columnas que vamos a usar en el TP
    columnas_necesarias = [
        "ANO4", "TRIMESTRE", "AGLOMERADO", "PONDERA",
        "CH04", "CH06", "ESTADO", "NIVEL_ED", 
        "P21", "P47T", "PP04B_COD", "PP04D_COD",
        "CAT_OCUP",  # quizas lo uso para el modelo predictivo
    ]

    for archivo in ruta.glob("*.txt*"):
        match = re.search(r'(\d{4})_T(\d)', archivo.name)
        if match:
            anio = int(match.group(1))
            trimestre = int(match.group(2))
            
            print(f"Cargando {archivo.name}...") # print para ver el progreso

            try:
                # columnas necesarias
                df = pd.read_csv(
                    archivo, 
                    sep=";", 
                    low_memory=False,
                    usecols=lambda c: c in columnas_necesarias
                )
            except Exception as e:
                print(f"Error leyendo {archivo.name}: {e}")
                continue
            
            # filtro por aglomerado
            df = df[df["AGLOMERADO"].isin([7, 8])]
            
            # variables de tiempo
            df = df.assign(ANIO_ARCHIVO=anio, TRIMESTRE_ARCHIVO=trimestre)

            lista_df.append(df)

    # Unimos todo en un solo DataFrame 
    datos = pd.concat(lista_df, ignore_index=True)

    # Mapeo de ciudades
    mapa_ciudades = {7: "Posadas", 8: "Gran Resistencia"}
    datos['Nombre_Ciudad'] = datos['AGLOMERADO'].map(mapa_ciudades)

    return datos
