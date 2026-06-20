import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from carga_de_datos import cargar_datos

# 1. Cargar los datos
datos = cargar_datos()

# ==========================================
# 2. FILTRO Y LIMPIEZA DE INGRESOS (P21)
# ==========================================
print("--- 1. Preparando la variable P21 (Ingreso Ocupación Principal) ---")

# Paso A: Nos quedamos SOLO con la población ocupada (ESTADO == 1)
ocupados = datos[datos['ESTADO'] == 1].copy()
print(f"Total de personas ocupadas en la muestra: {len(ocupados)}")

# Paso B: Forzamos la columna P21 a ser numérica
# errors='coerce' convierte cualquier texto/basura que haya en la base a NaN
ocupados['P21'] = pd.to_numeric(ocupados['P21'], errors='coerce')

# Paso C: Limpiamos los sueldos 0 o códigos de error (negativos)
ocupados['P21_limpio'] = ocupados['P21'].apply(lambda x: np.nan if x <= 0 else x)

nulos_p21 = ocupados['P21_limpio'].isna().sum()
print(f"Cantidad de ocupados que no declaran ingresos (pasados a NaN): {nulos_p21}")

# ==========================================
# 3. AJUSTE POR INFLACIÓN (IPC INDEC ACUMULADO)
# ==========================================
# Diccionario de inflación encadenada con base 1 en 2016
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

print("\n--- 2. Ajustando sueldos por inflación ---")

# Calculamos el multiplicador exacto para llevar todo a precios de 2025
# Fórmula: Índice del año actual (2025) / Índice del año de la encuesta
ocupados['Multiplicador_IPC'] = FACTORES[2025] / ocupados['ANIO_ARCHIVO'].map(FACTORES)

# Multiplicamos el sueldo nominal limpio por este factor para obtener el sueldo REAL
ocupados['P21_real'] = ocupados['P21_limpio'] * ocupados['Multiplicador_IPC']

# ==========================================
# 4. COMPROBACIÓN Y ESTADÍSTICAS DE INGRESOS
# ==========================================
print("\nCalculando medidas estadísticas de los salarios reales...")

# Agrupamos por Año y Ciudad para obtener la radiografía completa
resumen_ingresos = ocupados.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad'])['P21_real'].agg(
    Media='mean',
    Mediana='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    Maximo='max',          # Acá vamos a detectar exactamente en qué año están los outliers
    Dispersion_Std='std'   # Desviación estándar para medir la volatilidad salarial
).round(0)

# Le damos formato de moneda a la tabla de la consola para que sea fácil de leer
for col in resumen_ingresos.columns:
    resumen_ingresos[col] = resumen_ingresos[col].apply(lambda x: f"${x:,.0f}")

print("\n--- 3. Resumen Estadístico de Ingresos Reales (A precios de 2025) ---")
print(resumen_ingresos.to_string())

# ==========================================
# 4.B DETECCIÓN EXACTA DE OUTLIERS
# ==========================================
print("\nCalculando el umbral y la cantidad de valores atípicos (Outliers)...")

def calcular_outliers(serie):
    s = serie.dropna()
    if len(s) == 0:
        return pd.Series({'Limite_Superior': 0, 'Cant_Outliers': 0})
    
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    limite_sup = q3 + 1.5 * iqr 
    
    cant_outliers = (s > limite_sup).sum()
    
    return pd.Series({
        'Limite_Superior': limite_sup,
        'Cant_Outliers': cant_outliers
    })

tabla_outliers = ocupados.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad'])['P21_real'].apply(calcular_outliers).unstack()

tabla_outliers['Limite_Superior'] = tabla_outliers['Limite_Superior'].apply(lambda x: f"${x:,.0f}")
tabla_outliers['Cant_Outliers'] = tabla_outliers['Cant_Outliers'].astype(int)

print("\n--- 4. Reporte de Valores Atípicos por Año y Ciudad ---")
print(tabla_outliers.to_string())


# ==========================================
# 5. GENERACIÓN DEL GRÁFICO (BOXPLOT POR AÑO)
# ==========================================
print("\nGenerando Boxplot Histórico de Ingresos Reales...")

plt.figure(figsize=(15, 7)) # Lo hacemos más ancho para que entren los 10 años cómodos
sns.set_style("whitegrid")

# Propiedades para resaltar los valores atípicos (outliers)
propiedades_outliers = {
    "marker": "o", 
    "markerfacecolor": "#d62828", 
    "markeredgecolor": "black",
    "alpha": 0.5, 
    "markersize": 4
}

# Boxplot cruzando Año (X) e Ingreso Real (Y)
sns.boxplot(
    data=ocupados,
    x="ANIO_ARCHIVO",
    y="P21_real",
    hue="Nombre_Ciudad",
    palette={"Gran Resistencia": "#2a9d8f", "Posadas": "#e76f51"},
    flierprops=propiedades_outliers
)

plt.title("Evolución de la Distribución Salarial y Valores Atípicos (A precios de 2025)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Sueldo Mensual Real de la Ocupación Principal", fontsize=12)

# Le damos formato de dinero al eje Y
from matplotlib.ticker import FuncFormatter
formato_pesos = FuncFormatter(lambda x, pos: f'${x:,.0f}')
plt.gca().yaxis.set_major_formatter(formato_pesos)

# Movemos la leyenda afuera para que no tape los atípicos del último año
plt.legend(title="Aglomerado Urbano", bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig('boxplot_ingresos_historico.png', dpi=300, bbox_inches="tight")
print("Gráfico 'boxplot_ingresos_historico.png' generado y guardado con éxito.")

plt.show()