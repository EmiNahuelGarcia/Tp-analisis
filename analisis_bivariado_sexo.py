import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from carga_de_datos import cargar_datos

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
print("--- 1. Preparando datos para el análisis histórico (Ingresos vs. Sexo) ---")
datos = cargar_datos()

# Filtramos solo la población ocupada (ESTADO == 1)
ocupados = datos[datos['ESTADO'] == 1].copy()

# Mapeamos el Sexo (CH04)
ocupados["Sexo"] = ocupados["CH04"].map({1: "Varón", 2: "Mujer"})

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
print("\nCalculando estadísticas detalladas de la evolución salarial por sexo...")

# Agrupamos por Año, Ciudad y Sexo y calculamos todas las métricas clave
stats_historicas = ocupados.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad', 'Sexo'])['P21_real'].agg(
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
    stats_historicas[col] = stats_historicas[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "NaN")

# También calculamos la tabla de Brecha de Medianas para no perderla
evolucion_mediana = ocupados.groupby(['ANIO_ARCHIVO', 'Nombre_Ciudad', 'Sexo'])['P21_real'].median().unstack()
evolucion_mediana['Brecha (%)'] = ((evolucion_mediana['Varón'] - evolucion_mediana['Mujer']) / evolucion_mediana['Mujer'] * 100).round(1)
evolucion_mediana['Brecha (%)'] = evolucion_mediana['Brecha (%)'].apply(lambda x: f"{x}%")

print("\n--- Estadísticas Detalladas de Ingreso Real (A precios de 2025) ---")
print(stats_historicas.to_string())

print("\n--- Brecha Salarial Histórica (%) ---")
print(evolucion_mediana[['Brecha (%)']].to_string())


# ==========================================
# 3. GENERACIÓN DEL GRÁFICO DE LÍNEAS (PANEL DOBLE)
# ==========================================
print("\nGenerando Gráfico de Líneas Histórico...")

# Configuramos el lienzo con dos gráficos lado a lado (comparten el eje Y para que sean comparables)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.set_style("whitegrid")

paleta_sexo = {"Varón": "#457b9d", "Mujer": "#e63946"} # Azul para varón, Rojo para mujer

# --- Panel 1: Gran Resistencia ---
df_resistencia = ocupados[ocupados['Nombre_Ciudad'] == 'Gran Resistencia']
sns.lineplot(
    data=df_resistencia, x="ANIO_ARCHIVO", y="P21_real", hue="Sexo",
    estimator=np.median, errorbar=None, marker="o", linewidth=2.5, markersize=7,
    palette=paleta_sexo, ax=axes[0]
)
axes[0].set_title("Gran Resistencia", fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel("Año", fontsize=12)
# Eje Y actualizado para explicitar el año base
axes[0].set_ylabel("Mediana Salarial (ARS Constantes deflactados a 2025)", fontsize=12)
axes[0].set_xticks(range(2016, 2026))
axes[0].legend().set_visible(False) # Ocultamos la leyenda acá para no repetir

# --- Panel 2: Posadas ---
df_posadas = ocupados[ocupados['Nombre_Ciudad'] == 'Posadas']
sns.lineplot(
    data=df_posadas, x="ANIO_ARCHIVO", y="P21_real", hue="Sexo",
    estimator=np.median, errorbar=None, marker="o", linewidth=2.5, markersize=7,
    palette=paleta_sexo, ax=axes[1]
)
axes[1].set_title("Posadas", fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel("Año", fontsize=12)
axes[1].set_xticks(range(2016, 2026))

# Movemos la leyenda unificada afuera del gráfico derecho
axes[1].legend(title="Sexo", bbox_to_anchor=(1.05, 1), loc='upper left')

# Formateamos el eje Y para que muestre plata
formato_pesos = FuncFormatter(lambda x, pos: f'${x:,.0f}')
axes[0].yaxis.set_major_formatter(formato_pesos)

# Título actualizado con la aclaración del IPC (usamos \n para la segunda línea)
plt.suptitle("Evolución Histórica de la Brecha Salarial de Género (2016-2025)\nValores deflactados mediante IPC", fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()

# Guardamos
plt.savefig('evolucion_brecha_sexo.png', dpi=300, bbox_inches="tight")
print("Gráfico 'evolucion_brecha_sexo.png' generado y guardado con éxito.")

plt.show()