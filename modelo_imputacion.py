import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.formula.api as smf
from carga_de_datos import cargar_datos

# ==========================================
# 1. CARGA Y PREPARACIÓN DEL DATASET
# ==========================================
print("--- 1. Preparando el dataset para Econometría ---")
datos = cargar_datos()

# Filtramos ocupados
df = datos[datos['ESTADO'] == 1].copy()

# ==========================================
# A. LIMPIEZA DE VARIABLE OBJETIVO (Y)
# ==========================================

# Ingreso laboral
df['P21'] = pd.to_numeric(df['P21'], errors='coerce')

# Deflactación a pesos constantes de 2025
FACTORES = {
    2016: 1.0,
    2017: 1.248,
    2018: 1.842048,
    2019: 2.833069824,
    2020: 3.85580803,
    2021: 5.818414318,
    2022: 11.33427109,
    2023: 35.29492018,
    2024: 76.87233615,
    2025: 101.087122
}

df['Multiplicador_IPC'] = (
    FACTORES[2025] /
    df['ANIO_ARCHIVO'].map(FACTORES)
)

df['P21_real'] = (
    df['P21'] *
    df['Multiplicador_IPC']
)

# ==========================================
# B. CONSTRUCCIÓN DE VARIABLES EXPLICATIVAS
# ==========================================

# Sexo
# Varón = 0
# Mujer = 1
df['Es_Mujer'] = df['CH04'].apply(
    lambda x: 1 if x == 2 else 0
)

# Edad
df['Edad'] = df['CH06'].replace(-1, np.nan)

# Edad al cuadrado (Mincer)
df['Edad_Cuadrado'] = df['Edad'] ** 2

# Aglomerado
# Posadas = 1
# Gran Resistencia = 0
df['Es_Posadas'] = df['Nombre_Ciudad'].apply(
    lambda x: 1 if x == 'Posadas' else 0
)

# Categoría ocupacional
df['CAT_OCUP'] = pd.to_numeric(
    df['CAT_OCUP'],
    errors='coerce'
)

# Eliminamos NS/NR
df = df[df['CAT_OCUP'] != 9]

# Nivel educativo
df['NIVEL_ED'] = pd.to_numeric(
    df['NIVEL_ED'],
    errors='coerce'
)

# Eliminamos categorías inválidas
df = df[~df['NIVEL_ED'].isin([0, 9])]

# Variables necesarias
columnas_necesarias = [
    'P21',
    'P21_real',
    'Es_Mujer',
    'Edad',
    'Edad_Cuadrado',
    'NIVEL_ED',
    'CAT_OCUP',
    'Es_Posadas',
    'ANIO_ARCHIVO',   
    'Nombre_Ciudad'  
]

df_modelo = df[columnas_necesarias].dropna()

# ==========================================
# 2. SEPARACIÓN DE DATOS Y TRANSFORMACIÓN LOG
# ==========================================
print("\n--- 2. Separando conjuntos de datos ---")

# Casos válidos
df_validos = df_modelo[
    df_modelo['P21'] > 0
].copy()

# Casos a imputar
df_a_imputar = df_modelo[
    df_modelo['P21'] == -9
].copy()

print(f"Casos válidos para entrenar: {len(df_validos)}")
print(f"Casos a imputar: {len(df_a_imputar)}")

# Logaritmo del ingreso real
df_validos['Log_P21_real'] = np.log(
    df_validos['P21_real']
)

# Split 80/20
df_train, df_test = train_test_split(
    df_validos,
    test_size=0.20,
    random_state=42
)

# ==========================================
# ESTADÍSTICAS DESCRIPTIVAS
# ==========================================
print("\n--- Estadísticas descriptivas ---")
print(df_validos['P21_real'].describe())

print("\nDistribución por sexo")
print(df_validos['Es_Mujer'].value_counts())

print("\nDistribución por ciudad")
print(df_validos['Es_Posadas'].value_counts())

print("\nCategoría ocupacional")
print(df_validos['CAT_OCUP'].value_counts().sort_index())

print("\nNivel educativo")
print(df_validos['NIVEL_ED'].value_counts().sort_index())

# ==========================================
# 3. ENTRENAMIENTO DEL MODELO
# ==========================================
print("\n--- 3. Entrenando Ecuación de Mincer (OLS con Efectos Fijos Temporales) ---")

formula = """
Log_P21_real ~
Es_Mujer +
Edad +
Edad_Cuadrado +
Es_Posadas +
C(CAT_OCUP, Treatment(3)) +
C(NIVEL_ED, Treatment(1)) +
C(ANIO_ARCHIVO) 
"""

modelo = smf.ols(
    formula=formula,
    data=df_train
).fit()

# Tabla econométrica completa
print(modelo.summary())

# ==========================================
# TABLA DE COEFICIENTES
# ==========================================
tabla_coef = pd.DataFrame({
    'Coeficiente': modelo.params,
    'P-Value': modelo.pvalues,
    'Error Std': modelo.bse
})

print("\n--- Tabla de Coeficientes ---")
print(tabla_coef.round(4))
print("\n--- Efectos porcentuales ---")

efectos = pd.DataFrame({
    'Variable': modelo.params.index,
    'Efecto_%': (np.exp(modelo.params)-1)*100
})

print(efectos.round(2))

significativas = tabla_coef[
    tabla_coef['P-Value'] < 0.05
]

print("\n--- Variables Significativas (5%) ---")
print(significativas.round(4))

# ==========================================
# 4. EVALUACIÓN DEL MODELO
# ==========================================
print("\n--- 4. Evaluación del Rendimiento (20% de prueba) ---")

prediccion_log = modelo.predict(df_test)

# Volvemos a pesos
prediccion_pesos = np.exp(
    prediccion_log
)

# Métricas
r2_pesos = r2_score(
    df_test['P21_real'],
    prediccion_pesos
)

mae = mean_absolute_error(
    df_test['P21_real'],
    prediccion_pesos
)

print(
    f"R-cuadrado del modelo Logarítmico: "
    f"{modelo.rsquared:.3f}"
)

print(
    f"R-cuadrado al reconvertir a Pesos: "
    f"{r2_pesos:.3f}"
)

print(
    f"Error Absoluto Medio (MAE): "
    f"${mae:,.0f}"
)

mape = np.mean(
    np.abs(
        (
            df_test['P21_real']
            - prediccion_pesos
        )
        / df_test['P21_real']
    )
) * 100

print(f"MAPE: {mape:.2f}%")

# ==========================================
# 5. IMPUTACIÓN CON CORRECCIÓN DEL LOGARITMO
# ==========================================
print("\n--- 5. Imputando valores con Corrección  del Logaritmo (-9) ---")

# 1. Calculamos el Factor de Smearing de Duan con los residuos del modelo de entrenamiento
residuos = modelo.resid
factor_duan = np.mean(np.exp(residuos))

# 2. Predecimos en logaritmos
imputacion_log = modelo.predict(df_a_imputar)

# 3. Volvemos a pesos y multiplicamos por el factor de corrección
imputacion_pesos = np.exp(imputacion_log) * factor_duan

print(f"Factor de corrección de Duan aplicado: {factor_duan:.4f}")
print(f"Se imputaron exitosamente {len(imputacion_pesos)} salarios.")
print(f"Sueldo promedio imputado ajustado: ${imputacion_pesos.mean():,.0f}")

# ==========================================
# COMPARACIÓN OBSERVADOS VS IMPUTADOS
# ==========================================
print("\n--- Comparación salarios observados vs imputados ---")

print("\nObservados")
print(
    df_validos['P21_real']
    .describe()
)

print("\nImputados")
print(
    pd.Series(imputacion_pesos)
    .describe()
)

print(
    f"Se imputaron exitosamente "
    f"{len(imputacion_pesos)} salarios."
)

print(
    f"Sueldo promedio imputado: "
    f"${imputacion_pesos.mean():,.0f}"
)

print(
    f"Sueldo máximo imputado: "
    f"${imputacion_pesos.max():,.0f}"
)

# ==========================================
# EXPORTACIÓN DE RESULTADOS
# ==========================================

tabla_coef.to_excel(
    "tabla_coeficientes.xlsx"
)

df_a_imputar['Ingreso_Imputado'] = imputacion_pesos

df_a_imputar.to_excel(
    "salarios_imputados.xlsx",
    index=False
)

print("\nArchivos exportados:")
print("- tabla_coeficientes.xlsx")
print("- salarios_imputados.xlsx")
# ==========================================
# 6. TABLAS COMPARATIVAS (ESTILO IMPACTO)
# ==========================================
print("\n--- Generando Tablas Comparativas por Año y Aglomerado ---")

# 1. Unimos predicciones y datos reales para poder comparar
df_validos_comp = df_validos.copy()
df_validos_comp['Ingreso_Final'] = df_validos_comp['P21_real']
df_validos_comp['Es_Imputado'] = 0

df_imputados_comp = df_a_imputar.copy()
df_imputados_comp['Ingreso_Final'] = imputacion_pesos
df_imputados_comp['Es_Imputado'] = 1

# Base de datos completa
df_total = pd.concat([df_validos_comp, df_imputados_comp])

# Unimos las predicciones del set de prueba para calcular el R2 por año
df_test_comp = df_test.copy()
df_test_comp['Prediccion_Pesos'] = prediccion_pesos

resultados = []

# 2. Iteramos por año y ciudad para calcular las métricas
for anio in sorted(df_total['ANIO_ARCHIVO'].unique()):
    for ciudad in ['Gran Resistencia', 'Posadas']:
        
        # Filtramos datos
        filtro_total = (df_total['ANIO_ARCHIVO'] == anio) & (df_total['Nombre_Ciudad'] == ciudad)
        filtro_test = (df_test_comp['ANIO_ARCHIVO'] == anio) & (df_test_comp['Nombre_Ciudad'] == ciudad)
        
        sub_total = df_total[filtro_total]
        sub_test = df_test_comp[filtro_test]
        
        # Métricas
        imputados_count = sub_total['Es_Imputado'].sum()
        
        # R2 de ese año y ciudad en particular (si hay suficientes datos de prueba)
        r2_grupo = np.nan
        if len(sub_test) > 5:
            r2_grupo = r2_score(sub_test['P21_real'], sub_test['Prediccion_Pesos'])
            
        # Medias
        media_antes = sub_total[sub_total['Es_Imputado'] == 0]['Ingreso_Final'].mean()
        media_despues = sub_total['Ingreso_Final'].mean()
        
        # Variación porcentual
        variacion = ((media_despues - media_antes) / media_antes) * 100 if media_antes > 0 else 0
        
        resultados.append({
            'Año': int(anio),
            'Ciudad': ciudad,
            'Imputados': imputados_count,
            'R2': r2_grupo,
            'Media_Antes': media_antes,
            'Media_Despues': media_despues,
            'Variacion_%': variacion
        })

df_res = pd.DataFrame(resultados)

# ==========================================
# IMPRESIÓN DE TABLA 1 (Imputaciones y R2)
# ==========================================
resistencia = df_res[df_res['Ciudad'] == 'Gran Resistencia'].set_index('Año')[['Imputados', 'R2', 'Media_Antes', 'Media_Despues']]
posadas = df_res[df_res['Ciudad'] == 'Posadas'].set_index('Año')[['Imputados', 'R2', 'Media_Antes', 'Media_Despues']]

resistencia.columns = ['Imp_GR', 'R2_GR', 'Antes_GR', 'Desp_GR']
posadas.columns = ['Imp_POS', 'R2_POS', 'Antes_POS', 'Desp_POS']

tabla1 = pd.concat([resistencia, posadas], axis=1)

# Creamos las funciones lambda para formatear correctamente los números
formato1 = {
    'Imp_GR': lambda x: f"{x:.0f}", 
    'R2_GR': lambda x: f"{x:.3f}" if pd.notnull(x) else "-", 
    'Antes_GR': lambda x: f"${x:,.0f}", 
    'Desp_GR': lambda x: f"${x:,.0f}",
    'Imp_POS': lambda x: f"{x:.0f}", 
    'R2_POS': lambda x: f"{x:.3f}" if pd.notnull(x) else "-", 
    'Antes_POS': lambda x: f"${x:,.0f}", 
    'Desp_POS': lambda x: f"${x:,.0f}"
}

print("\n--- Tabla 1: Resumen de Imputación y Rendimiento por Aglomerado ---")
print("Año |   Gran Resistencia (Imputados, R2, Antes, Despues)   |       Posadas (Imputados, R2, Antes, Despues)")
print(tabla1.to_string(formatters=formato1))

# ==========================================
# IMPRESIÓN DE TABLA 2 (Variaciones)
# ==========================================
var_gr = df_res[df_res['Ciudad'] == 'Gran Resistencia'].set_index('Año')[['Media_Antes', 'Media_Despues', 'Variacion_%']]
var_pos = df_res[df_res['Ciudad'] == 'Posadas'].set_index('Año')[['Media_Antes', 'Media_Despues', 'Variacion_%']]

var_gr.columns = ['Antes_GR', 'Desp_GR', 'Var_GR']
var_pos.columns = ['Antes_POS', 'Desp_POS', 'Var_POS']

tabla2 = pd.concat([var_gr, var_pos], axis=1)

# Funciones lambda para los porcentajes y dinero
formato2 = {
    'Antes_GR': lambda x: f"${x:,.0f}", 
    'Desp_GR': lambda x: f"${x:,.0f}", 
    'Var_GR': lambda x: f"{x:.2f}%",
    'Antes_POS': lambda x: f"${x:,.0f}", 
    'Desp_POS': lambda x: f"${x:,.0f}", 
    'Var_POS': lambda x: f"{x:.2f}%"
}

print("\n--- Tabla 2: Impacto de la Imputación sobre las Medias Salariales ---")
print("Año |    Gran Resistencia (Antes, Despues, Variación)    |         Posadas (Antes, Despues, Variación)")
print(tabla2.to_string(formatters=formato2))