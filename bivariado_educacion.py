import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from carga_de_datos import cargar_datos

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
print("--- 1. Preparando datos para el análisis histórico (Ingresos vs. Educación) ---")
datos = cargar_datos()

# Filtramos solo la población ocupada (ESTADO == 1)
ocupados = datos[datos['ESTADO'] == 1].copy()

# Mapeamos el Nivel Educativo (NIVEL_ED) descartando 0 y 9 (Sin respuesta)
ocupados['NIVEL_ED_limpio'] = ocupados['NIVEL_ED'].replace([0, 9], np.nan)
mapa_educacion = {
    7: "0. Sin instrucción",
    1: "1. Prim. Incompleta",
    2: "2. Prim. Completa",
    3: "3. Sec. Incompleta",
    4: "4. Sec. Completa",
    5: "5. Sup. Incompleta",
    6: "6. Sup. Completa"
}
ocupados["Nivel_Educativo"] = ocupados["NIVEL_ED_limpio"].map(mapa_educacion)

# Limpiamos los Ingresos y ajustamos por inflación a valores de 2025
ocupados['P21_numeric'] = pd.to_numeric(ocupados['P21'], errors='coerce')
ocupados['P21_limpio'] = ocupados['P21_numeric'].apply(lambda x: np.nan if x <= 0 else x)

FACTORES = {
    2016: 1.0, 2017: 1.248, 2018: 1.842048, 2019: 2.833069824, 2020: 3.85580803,
    2021: 5.818414318, 2022: 11.33427109, 2023: 35.29492018, 2024: 76.87233615, 2025: 101.087122
}
ocupados['Multiplicador_IPC'] = FACTORES[2025] / ocupados['ANIO_ARCHIVO'].map(FACTORES)
ocupados['P21_real'] = ocupados['P21_limpio'] * ocupados['Multiplicador_IPC']

# ==========================================
# 2. CÁLCULO DE LA TABLA HISTÓRICA DETALLADA
# ==========================================
print("\nCalculando estadísticas detalladas de la evolución salarial por Nivel Educativo...")

# Agrupamos por Año, Ciudad y Educación y calculamos todas las métricas clave
stats_ed = ocupados.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad', 'Nivel_Educativo'])['P21_real'].agg(
    Casos='count',
    Media='mean',
    Mediana='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    Desvio_Std='std'
).round(0)

# Formateamos las columnas de dinero para que se lean bien en la consola
cols_dinero = ['Media', 'Mediana', 'Q1', 'Q3', 'Desvio_Std']
for col in cols_dinero:
    stats_ed[col] = stats_ed[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "NaN")

print("\n--- Estadísticas Detalladas de Ingreso Real según Educación (A precios de 2025) ---")
print(stats_ed.to_string())

# ==========================================
# 3. GENERACIÓN DEL GRÁFICO DE LÍNEAS (PANEL DOBLE)
# ==========================================
print("\nGenerando Gráfico de Líneas Histórico (basado en Medianas)...")

# Configuramos el lienzo
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
sns.set_style("whitegrid")

# Usamos una paleta secuencial
paleta_ed = sns.color_palette("rocket_r", n_colors=7)

# Definimos el orden estricto para que Seaborn no mezcle los colores
orden_educacion = [
    "0. Sin instrucción", "1. Prim. Incompleta", "2. Prim. Completa",
    "3. Sec. Incompleta", "4. Sec. Completa", "5. Sup. Incompleta", "6. Sup. Completa"
]

# --- Panel 1: Gran Resistencia ---
df_resistencia = ocupados[ocupados['Nombre_Ciudad'] == 'Gran Resistencia'].dropna(subset=['Nivel_Educativo'])
sns.lineplot(
    data=df_resistencia, x="ANIO_ARCHIVO", y="P21_real", hue="Nivel_Educativo",
    hue_order=orden_educacion, # <--- Forzamos el orden de los colores
    estimator=np.median, errorbar=None, marker="o", linewidth=2.5, markersize=6,
    palette=paleta_ed, ax=axes[0]
)
axes[0].set_title("Gran Resistencia", fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel("Año", fontsize=12)
axes[0].set_ylabel("Mediana Salarial (ARS Constantes deflactados a 2025)", fontsize=12)
axes[0].set_xticks(range(2016, 2026))
axes[0].legend().set_visible(False)

# --- Panel 2: Posadas ---
df_posadas = ocupados[ocupados['Nombre_Ciudad'] == 'Posadas'].dropna(subset=['Nivel_Educativo'])
sns.lineplot(
    data=df_posadas, x="ANIO_ARCHIVO", y="P21_real", hue="Nivel_Educativo",
    hue_order=orden_educacion, # <--- Forzamos el orden de los colores
    estimator=np.median, errorbar=None, marker="o", linewidth=2.5, markersize=6,
    palette=paleta_ed, ax=axes[1]
)
axes[1].set_title("Posadas", fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel("Año", fontsize=12)
axes[1].set_xticks(range(2016, 2026))

# --- ARREGLO DE LA LEYENDA ---
# Extraemos las etiquetas y los colores del gráfico
handles, labels = axes[1].get_legend_handles_labels()
# Los invertimos ([::-1]) para que "6. Sup. Completa" quede jerarquizada arriba de todo
axes[1].legend(handles[::-1], labels[::-1], title="Máximo Nivel Educativo", bbox_to_anchor=(1.05, 1), loc='upper left')

# Formateamos el eje Y para mostrar valores monetarios
formato_pesos = FuncFormatter(lambda x, pos: f'${x:,.0f}')
axes[0].yaxis.set_major_formatter(formato_pesos)

plt.suptitle("Evolución Histórica de la Mediana Salarial según Nivel Educativo (2016-2025)\nValores deflactados mediante IPC", fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()

# Guardamos
plt.savefig('evolucion_mediana_educacion.png', dpi=300, bbox_inches="tight")
print("Gráfico 'evolucion_mediana_educacion.png' generado y guardado con éxito.")

plt.show()