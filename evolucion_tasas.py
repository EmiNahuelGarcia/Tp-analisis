import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from carga_de_datos import cargar_datos

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
print("--- 1. Preparando datos para el cálculo de Tasas Oficiales ---")
datos = cargar_datos()

# Limpiamos y mapeamos el estado (actividad)
datos['ESTADO_limpio'] = datos['ESTADO'].replace(0, np.nan)
mapa_estado = {
    1: "Ocupado", 
    2: "Desocupado", 
    3: "Inactivo", 
    4: "Menor de 10 años"
}
datos["Condicion_Actividad"] = datos["ESTADO_limpio"].map(mapa_estado)

# ==========================================
# 2. CÁLCULO DE TASAS OFICIALES (Metodología INDEC)
# ==========================================
print("\nCalculando promedios anuales y tasas (Actividad, Empleo, Desocupación)...")
datos_validos = datos.dropna(subset=['Condicion_Actividad'])

# Promedios anuales de los factores de expansión (PONDERA)
actividad_trimestral = datos_validos.groupby(
    ["ANIO_ARCHIVO", "TRIMESTRE_ARCHIVO", "Nombre_Ciudad", "Condicion_Actividad"]
)["PONDERA"].sum().reset_index()

actividad_anual = actividad_trimestral.groupby(
    ["ANIO_ARCHIVO", "Nombre_Ciudad", "Condicion_Actividad"]
)["PONDERA"].mean().unstack(fill_value=0)

# Población Total y Población Económicamente Activa (PEA)
actividad_anual["Poblacion_Total"] = actividad_anual.sum(axis=1)
actividad_anual["PEA"] = actividad_anual["Ocupado"] + actividad_anual["Desocupado"]

# Cálculo de las Tasas
tasas = pd.DataFrame(index=actividad_anual.index)
tasas["Tasa_Actividad"] = (actividad_anual["PEA"] / actividad_anual["Poblacion_Total"]) * 100
tasas["Tasa_Empleo"] = (actividad_anual["Ocupado"] / actividad_anual["Poblacion_Total"]) * 100
tasas["Tasa_Desocupacion"] = (actividad_anual["Desocupado"] / actividad_anual["PEA"]) * 100

tabla_tasas = tasas.round(1)

print("\n--- Evolución de Tasas Oficiales del Mercado Laboral (%) ---")
print(tabla_tasas.to_string())

# ==========================================
# 3. GENERACIÓN DEL GRÁFICO (LÍNEAS)
# ==========================================
print("\nGenerando Gráfico de Líneas Histórico...")
df_tasas = tabla_tasas.reset_index()

# Creamos 3 paneles verticales, compartiendo el eje X (sharex=True) para mayor limpieza
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
sns.set_style("whitegrid")
paleta_ciudades = {"Gran Resistencia": "#2a9d8f", "Posadas": "#e76f51"}

# PANEL 1: Tasa de Actividad
sns.lineplot(
    data=df_tasas, x="ANIO_ARCHIVO", y="Tasa_Actividad", hue="Nombre_Ciudad", 
    marker="o", linewidth=2.5, markersize=8, palette=paleta_ciudades, ax=axes[0]
)
axes[0].set_title("1. Evolución Tasa de Actividad", fontsize=14, fontweight='bold', pad=10)
axes[0].set_ylabel("Porcentaje (%)", fontsize=11)
axes[0].legend(title="Aglomerado", bbox_to_anchor=(1.01, 1), loc='upper left')

# PANEL 2: Tasa de Empleo
sns.lineplot(
    data=df_tasas, x="ANIO_ARCHIVO", y="Tasa_Empleo", hue="Nombre_Ciudad", 
    marker="o", linewidth=2.5, markersize=8, palette=paleta_ciudades, ax=axes[1]
)
axes[1].set_title("2. Evolución Tasa de Empleo", fontsize=14, fontweight='bold', pad=10)
axes[1].set_ylabel("Porcentaje (%)", fontsize=11)
axes[1].get_legend().remove()

# PANEL 3: Tasa de Desocupación
sns.lineplot(
    data=df_tasas, x="ANIO_ARCHIVO", y="Tasa_Desocupacion", hue="Nombre_Ciudad", 
    marker="o", linewidth=2.5, markersize=8, palette=paleta_ciudades, ax=axes[2]
)
axes[2].set_title("3. Evolución Tasa de Desocupación", fontsize=14, fontweight='bold', pad=10)
axes[2].set_ylabel("Porcentaje (%)", fontsize=11)
axes[2].set_xlabel("Año", fontsize=12)
axes[2].set_xticks(range(2016, 2026))
axes[2].get_legend().remove()

plt.suptitle("Evolución de las Tasas Oficiales del Mercado Laboral (2016-2025)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig("evolucion_tasas_lineas.png", dpi=300, bbox_inches="tight")
print("Gráfico 'evolucion_tasas_lineas.png' generado y guardado con éxito.")
plt.show()