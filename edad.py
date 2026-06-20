import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from carga_de_datos import cargar_datos

# 1. Cargar los datos
datos = cargar_datos()

# ==========================================
# 2. LIMPIEZA Y COMPROBACIÓN DE CALIDAD
# ==========================================
print("--- 1. Comprobación de la variable CH06 (Edad) ---")

# En la EPH, los bebés menores de 1 año (0 años cumplidos) se registran como -1.
# Los transformamos en 0 para que los cálculos (Media, Mediana) sean matemáticamente correctos.
datos['EDAD_limpia'] = datos['CH06'].replace(-1, 0)

nulos_edad = datos['EDAD_limpia'].isna().sum()
print(f"Cantidad de valores nulos pasados a NaN: {nulos_edad}")
print(f"Edad mínima registrada: {datos['EDAD_limpia'].min()} años")
print(f"Edad máxima registrada (valor atípico): {datos['EDAD_limpia'].max()} años")

# ==========================================
# 3. EVOLUCIÓN HISTÓRICA: TENDENCIA CENTRAL Y POSICIÓN
# ==========================================
print("\nCalculando medidas estadísticas de la edad...")

# Agrupamos por Año y Ciudad, y aplicamos múltiples funciones estadísticas a la vez
resumen_edad = datos.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad'])['EDAD_limpia'].agg(
    Media='mean',
    Mediana='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    Minimo='min',
    Maximo='max'
).round(1) # Redondeamos a 1 decimal para que sea fácil de leer

print("\n--- 2. Resumen Estadístico de Edad por Año y Ciudad ---")
print(resumen_edad)

# ==========================================
# 4. GENERACIÓN DEL GRÁFICO (BOXPLOT FINAL)
# ==========================================
plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")

# Propiedades para que los outliers se vean como puntos rojos y notorios
propiedades_outliers = {
    "marker": "o", 
    "markerfacecolor": "#d62828", 
    "markeredgecolor": "black",
    "alpha": 0.6,
    "markersize": 5
}

# Generamos el boxplot
sns.boxplot(
    data=datos, 
    x="ANIO_ARCHIVO", 
    y="EDAD_limpia", 
    hue="Nombre_Ciudad", 
    palette={"Gran Resistencia": "#2a9d8f", "Posadas": "#e76f51"},
    whis=1.5,                       
    flierprops=propiedades_outliers 
)

plt.title("Distribución de edades por año y ciudad", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Edad", fontsize=12)

# Límites y marcas del eje Y mejorados
plt.ylim(0, 105)              
plt.yticks(range(0, 106, 10)) 

# Movemos la leyenda afuera para que no tape las cajas del año 2025
plt.legend(title="Ciudad", bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig("grafico_edad_final.png", dpi=300, bbox_inches="tight")
print("\nGráfico 'grafico_edad_final.png' generado y guardado con éxito.")

plt.show()

# ==========================================
# 5. GENERACIÓN DEL GRÁFICO (BARRAS DE EDAD PROMEDIO)
# ==========================================
print("\nGenerando Gráfico de Barras de Edad Promedio...")

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# El barplot automáticamente calcula el promedio (media) de la edad para cada año
sns.barplot(
    data=datos,
    x="ANIO_ARCHIVO",
    y="EDAD_limpia",
    hue="Nombre_Ciudad",
    palette={"Gran Resistencia": "#2a9d8f", "Posadas": "#e76f51"},
    errorbar=None # Quitamos las líneas de error para que la barra quede limpia
)

# Títulos y etiquetas (Ahora sí, Año abajo y Edad a la izquierda)
plt.title("Evolución de la Edad Promedio por Aglomerado (2016-2025)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Edad Promedio (Años)", fontsize=12)

# Fijamos el límite del eje Y un poco por encima del promedio máximo para que se lea bien
plt.ylim(0, 45)

# Movemos la leyenda afuera
plt.legend(title="Aglomerado", bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig('barras_edad_promedio.png', dpi=300, bbox_inches="tight")
print("Gráfico 'barras_edad_promedio.png' generado y guardado con éxito.")

plt.show()



