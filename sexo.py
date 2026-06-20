import pandas as pd
from carga_de_datos import cargar_datos
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
datos = cargar_datos()

# 2. Crear la columna 'Sexo'
mapa_sexo = {1: "Varón", 2: "Mujer"}
datos["Sexo"] = datos["CH04"].map(mapa_sexo)

# 3. Validar NaNs y valores únicos
cantidad_nulos = datos["CH04"].isna().sum()
print(f"Cantidad de valores NaN explícitos en CH04: {cantidad_nulos}")

valores_unicos = datos["CH04"].value_counts(dropna=False)
print("\nTodos los valores encontrados en la columna CH04 (Sexo):")
print(valores_unicos)

# 4. Agrupar por año, ciudad y sexo usando la SUMA de ponderadores
poblacion_anual_total = datos.groupby(
    ["ANIO_ARCHIVO", "Nombre_Ciudad", "Sexo"]
)["PONDERA"].sum()

# 5. Calcular porcentajes sobre el total anual
porcentajes_anuales = (poblacion_anual_total /
                       poblacion_anual_total.groupby(level=["ANIO_ARCHIVO", "Nombre_Ciudad"]).transform("sum")) * 100

tabla_final = porcentajes_anuales.unstack().round(1)

print("\n--- Porcentaje Anual de población por Sexo y Ciudad ---")
print(tabla_final)


#GRAFICO

# 1. Aplanamos la tabla
df_grafico = tabla_final.reset_index()

# 2. "Derretimos" (melt) la tabla para agrupar las columnas Mujer y Varón en una sola
df_melted = df_grafico.melt(
    id_vars=["ANIO_ARCHIVO", "Nombre_Ciudad"], 
    value_vars=["Mujer", "Varón"],
    var_name="Sexo",
    value_name="Porcentaje"
)

# =====================================================================
# EL TRUCO: Creamos una categoría nueva que combine Ciudad y Sexo
df_melted["Ciudad_y_Sexo"] = df_melted["Nombre_Ciudad"] + " - " + df_melted["Sexo"]
# =====================================================================

# 3. Configuramos el gráfico
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# 4. Armamos el gráfico de líneas con 4 colores distintos
# Pasamos la nueva columna "Ciudad_y_Sexo" al hue (color)
sns.lineplot(
    data=df_melted, 
    x="ANIO_ARCHIVO", 
    y="Porcentaje", 
    hue="Ciudad_y_Sexo", # Esto genera los 4 colores distintos
    marker="o",          # Forzamos a que TODOS los puntos sean círculos (adiós cruces)
    linewidth=2.5,
    markersize=8,
    palette="Set1"       # Usamos una paleta de colores fuertes y distinguibles
)

# 5. Etiquetas y título
plt.title("Distribución Poblacional por Sexo y Ciudad (2016-2025)", fontsize=15, pad=15)
plt.ylabel("Porcentaje de la Población (%)", fontsize=12)
plt.xlabel("Año", fontsize=12)

# 6. Ajustes finos: Rango del eje Y para que se vean bien separadas las líneas de 48% y 52%
plt.ylim(45, 55) 
plt.xticks(df_melted["ANIO_ARCHIVO"].unique())

# 7. Movemos la leyenda afuera del gráfico para que no tape las líneas
plt.legend(title="Ciudad y Sexo", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout() # Ajusta los márgenes automáticamente

# 8. Guardamos y mostramos
plt.savefig("grafico_sexo_4colores.png", dpi=300)
print("Gráfico 'grafico_sexo_4colores.png' generado y guardado con éxito.")
plt.show()