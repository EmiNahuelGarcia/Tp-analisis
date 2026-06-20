import pandas as pd
import numpy as np
from carga_de_datos import cargar_datos

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
print("Procesando base de datos con reglas estrictas del INDEC...\n")
datos = cargar_datos()

# --- PREPARACIÓN CH06 (EDAD) ---
datos['EDAD_limpia'] = datos['CH06'].replace(-1, 0)

# --- PREPARACIÓN P21 (INGRESOS SOLO OCUPADOS) ---
ocupados = datos[datos['ESTADO'] == 1].copy()
ocupados['P21_numeric'] = pd.to_numeric(ocupados['P21'], errors='coerce')

# Ajuste por inflación para Atípicos
FACTORES = {
    2016: 1.0, 2017: 1.248, 2018: 1.842048, 2019: 2.833069824, 2020: 3.85580803,
    2021: 5.818414318, 2022: 11.33427109, 2023: 35.29492018, 2024: 76.87233615, 2025: 101.087122
}
ocupados['Multiplicador_IPC'] = FACTORES[2025] / ocupados['ANIO_ARCHIVO'].map(FACTORES)

# P21 Limpio para Outliers: Excluimos <= 0 (incluyendo el -9)
ocupados['P21_limpio'] = ocupados['P21_numeric'].apply(lambda x: np.nan if x <= 0 else x)
ocupados['P21_real'] = ocupados['P21_limpio'] * ocupados['Multiplicador_IPC']

# Función para calcular outliers (IQR)
def contar_outliers(serie):
    s = serie.dropna()
    if len(s) == 0: return 0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    limite_sup = q3 + 1.5 * (q3 - q1)
    return (s > limite_sup).sum()

outliers_p21_por_año = ocupados.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad'])['P21_real'].apply(contar_outliers).unstack()

# ==========================================
# 2. CONSTRUCCIÓN DE LA TABLA RESUMEN
# ==========================================
ciudades = ['Posadas', 'Gran Resistencia']
resultados = {'Posadas': [], 'Gran Resistencia': []}

for ciudad in ciudades:
    df_c = datos[datos['Nombre_Ciudad'] == ciudad]
    df_o = ocupados[ocupados['Nombre_Ciudad'] == ciudad]
    
    # --- CH04: Sexo ---
    # Faltantes: Nulos o cualquier cosa que no sea 1 o 2
    ch04_falt = df_c['CH04'].isna().sum() + (~df_c['CH04'].isin([1, 2])).sum()
    ch04_nr = 0 # No hay código específico NS/NC documentado distinto a faltante
    ch04_out = "-"
    
    # --- ESTADO: Actividad ---
    # No respuesta: 0 (Entrevista no realizada)
    est_nr = (df_c['ESTADO'] == 0).sum()
    # Faltantes: Nulos o cualquier cosa que no sea 0, 1, 2, 3, 4
    est_falt = df_c['ESTADO'].isna().sum() + (~df_c['ESTADO'].isin([0, 1, 2, 3, 4])).sum()
    est_out = "-"
    
    # --- NIVEL_ED: Educativo ---
    # No respuesta: 9
    ned_nr = (df_c['NIVEL_ED'] == 9).sum()
    # Faltantes: Nulos o cualquier cosa que no sea 1,2,3,4,5,6,7,9
    ned_falt = df_c['NIVEL_ED'].isna().sum() + (~df_c['NIVEL_ED'].isin([1, 2, 3, 4, 5, 6, 7, 9])).sum()
    ned_out = "-"
    
    # --- CH06: Edad ---
    # No respuesta: 999 (Asumiendo 99 como válido)
    ch06_nr = (df_c['CH06'] == 999).sum()
    # Faltantes: Nulos o menores a -1 (códigos corruptos)
    ch06_falt = df_c['CH06'].isna().sum() + (df_c['CH06'] < -1).sum()
    # Atípicos: Excluimos el código de NS/NC si existiera para la cuenta
    edad_valida = df_c[df_c['CH06'] != 999]['EDAD_limpia']
    ch06_out = contar_outliers(edad_valida)
    
    # --- P21: Ingreso Principal (Solo Ocupados) ---
    # No respuesta: -9
    p21_nr = (df_o['P21_numeric'] == -9).sum()
    # Faltantes: Nulos + Valores <= 0 (que no sean el -9 de NR)
    p21_falt = df_o['P21_numeric'].isna().sum() + ((df_o['P21_numeric'] <= 0) & (df_o['P21_numeric'] != -9)).sum()
    # Atípicos
    p21_out = int(outliers_p21_por_año[ciudad].sum()) if ciudad in outliers_p21_por_año.columns else 0

    # Guardado de columna
    resultados[ciudad] = [
        ch04_falt, ch04_nr, ch04_out,
        est_falt, est_nr, est_out,
        ned_falt, ned_nr, ned_out,
        ch06_falt, ch06_nr, ch06_out,
        p21_falt, p21_nr, p21_out
    ]

# ==========================================
# 3. IMPRESIÓN CON FORMATO TABULAR
# ==========================================
nombres_filas = [
    'CH04 - Faltantes', 'CH04 - No respuesta', 'CH04 - Atípicos',
    'ESTADO - Faltantes', 'ESTADO - No respuesta', 'ESTADO - Atípicos',
    'NIVEL ED - Faltantes', 'NIVEL ED - No respuesta', 'NIVEL ED - Atípicos',
    'CH06 - Faltantes', 'CH06 - No respuesta', 'CH06 - Atípicos',
    'P21 - Faltantes', 'P21 - No respuesta', 'P21 - Atípicos'
]

tabla_final = pd.DataFrame(resultados, index=nombres_filas)

print("=============================================")
print(" 1.2. No respuesta y Atípicos (Metadatos)")
print("=============================================")
print(tabla_final.to_string())
print("=============================================\n")