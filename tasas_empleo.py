import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from carga_de_datos import cargar_datos

# 1. Cargar los datos
datos = cargar_datos()

# ==========================================
# 2. LIMPIEZA Y COMPROBACIÓN DE CALIDAD DE DATOS
# ==========================================
# Limpiamos el 0 (Entrevista no realizada) pasándolo a NaN
datos['ESTADO_limpio'] = datos['ESTADO'].replace(0, np.nan)

print("--- 1. Comprobación de la variable ESTADO ---")
# Contamos cuántos ceros (ahora NaN) había realmente
cantidad_nulos = datos['ESTADO_limpio'].isna().sum()
print(f"Cantidad de entrevistas no realizadas (código 0 pasado a NaN): {cantidad_nulos}")

# Verificamos que ya no exista el 0 en los datos limpios
print("\nTodos los valores válidos en la columna ESTADO_limpio:")
print(datos["ESTADO_limpio"].value_counts(dropna=False).sort_index())

# ==========================================
# 3. CREAR LA COLUMNA 'Condicion_Actividad'
# ==========================================
# Mapeo según el manual de la EPH
mapa_estado = {
    1: "Ocupado", 
    2: "Desocupado", 
    3: "Inactivo", 
    4: "Menor de 10 años"
}
datos["Condicion_Actividad"] = datos["ESTADO_limpio"].map(mapa_estado)

# ==========================================
# 4. CÁLCULO DE TASAS OFICIALES (INDEC)
# ==========================================
# Filtramos los nulos para hacer el cálculo matemáticamente exacto
datos_validos = datos.dropna(subset=['Condicion_Actividad'])

# Paso A: La "foto" trimestral sin duplicar
actividad_trimestral = datos_validos.groupby(
    ["ANIO_ARCHIVO", "TRIMESTRE_ARCHIVO", "Nombre_Ciudad", "Condicion_Actividad"]
)["PONDERA"].sum().reset_index()

# Paso B: Promediamos los trimestres y pasamos las categorías a columnas (unstack)
actividad_anual = actividad_trimestral.groupby(
    ["ANIO_ARCHIVO", "Nombre_Ciudad", "Condicion_Actividad"]
)["PONDERA"].mean().unstack(fill_value=0)

# Paso C: Calculamos los totales poblacionales base
actividad_anual["Poblacion_Total"] = actividad_anual.sum(axis=1)
actividad_anual["PEA"] = actividad_anual["Ocupado"] + actividad_anual["Desocupado"]
# Redondeamos a enteros porque son personas ponderadas
pea_anual = actividad_anual["PEA"].round(0)

print("\n--- PEA ponderada anual por ciudad ---")
print(pea_anual)

# Paso D: Cálculo de Tasas Oficiales INDEC
tasas = pd.DataFrame(index=actividad_anual.index)
tasas["Tasa_Actividad"] = (actividad_anual["PEA"] / actividad_anual["Poblacion_Total"]) * 100
tasas["Tasa_Empleo"] = (actividad_anual["Ocupado"] / actividad_anual["Poblacion_Total"]) * 100
tasas["Tasa_Desocupacion"] = (actividad_anual["Desocupado"] / actividad_anual["PEA"]) * 100

# Redondeamos a 1 decimal
tabla_tasas = tasas.round(1)

print("\n--- 2. Tasas Oficiales del Mercado Laboral (Promedio Anual) ---")
print(tabla_tasas)

# ==========================================
# 5. GENERACIÓN DEL GRÁFICO (3 PANELES MEJORADOS)
# ==========================================
# Aplanamos la tabla para Seaborn
df_tasas = tabla_tasas.reset_index()

# Configuramos el lienzo: aumentamos el alto a 15 y sacamos sharex=True 
# para que los años se repitan en cada gráfico
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
sns.set_style("whitegrid")

# Paleta de colores fija para las ciudades
paleta_ciudades = {"Gran Resistencia": "#2a9d8f", "Posadas": "#e76f51"}

# PANEL 1: Tasa de Actividad
sns.barplot(data=df_tasas, x="ANIO_ARCHIVO", y="Tasa_Actividad", hue="Nombre_Ciudad", ax=axes[0], palette=paleta_ciudades)
axes[0].set_title("1. Tasa de Actividad", fontsize=12, fontweight='bold', pad=10)
axes[0].set_ylabel("Porcentaje (%)")
axes[0].set_ylim(35, 55) 
axes[0].legend(title="Aglomerado", bbox_to_anchor=(1.05, 1), loc='upper left')

# PANEL 2: Tasa de Empleo
sns.barplot(data=df_tasas, x="ANIO_ARCHIVO", y="Tasa_Empleo", hue="Nombre_Ciudad", ax=axes[1], palette=paleta_ciudades)
axes[1].set_title("2. Tasa de Empleo", fontsize=12, fontweight='bold', pad=10)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_ylim(30, 52) # Subimos el techo para que entren las etiquetas
axes[1].get_legend().remove() 

# PANEL 3: Tasa de Desocupación
sns.barplot(data=df_tasas, x="ANIO_ARCHIVO", y="Tasa_Desocupacion", hue="Nombre_Ciudad", ax=axes[2], palette=paleta_ciudades)
axes[2].set_title("3. Tasa de Desocupación", fontsize=12, fontweight='bold', pad=10)
axes[2].set_ylabel("Porcentaje (%)")
axes[2].set_ylim(0, 11) # Subimos el techo para el pico de desocupación
axes[2].get_legend().remove()

# ====================================================================
# --- LO NUEVO: ETIQUETAS Y AÑOS EN TODOS LOS PANELES ---
for ax in axes:
    ax.set_xlabel("Año", fontsize=10)
    # Iteramos sobre las barras de Posadas y Gran Resistencia
    for c in ax.containers:
        # Armamos el texto con el porcentaje y lo ubicamos arriba de la barra (padding=3)
        etiquetas = [f'{v.get_height():.1f}%' for v in c]
        ax.bar_label(c, labels=etiquetas, padding=3, fontsize=9, fontweight='bold', color='#333333')
# ====================================================================

# Ajustes finales y guardado
plt.suptitle("Evolución de las Tasas Oficiales del Mercado Laboral (2016-2025)", fontsize=16, y=1.02)
plt.tight_layout(h_pad=4.0)

plt.savefig("grafico_tasas_oficiales_mejorado.png", dpi=300, bbox_inches="tight")
print("\nGráfico 'grafico_tasas_oficiales_mejorado.png' generado y guardado con éxito.")

plt.show()