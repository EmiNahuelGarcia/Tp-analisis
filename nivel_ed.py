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
print("--- Valores crudos de NIVEL_ED (Base original) ---")
print(datos['NIVEL_ED'].value_counts(dropna=False).sort_index())
print("--- 1. Comprobación de la variable NIVEL_ED ---")
# En la EPH, el código 9 significa "No Sabe / No Responde". Lo pasamos a NaN.
# También verificamos si hay algún 0 perdido.
datos['NIVEL_ED_limpio'] = datos['NIVEL_ED'].replace([0, 9], np.nan)

cantidad_nulos = datos['NIVEL_ED_limpio'].isna().sum()
print(f"Cantidad de valores nulos o sin respuesta (códigos 0 y 9 pasados a NaN): {cantidad_nulos}")
print("\nTodos los valores válidos en la columna NIVEL_ED_limpio:")
print(datos["NIVEL_ED_limpio"].value_counts(dropna=False).sort_index())

# ==========================================
# 3. CREAR LA COLUMNA ORDENADA
# ==========================================
# IMPORTANTE: Le ponemos un número adelante al texto para que Pandas y el gráfico 
# respeten el orden jerárquico natural (del 0 al 6) y no los ordenen alfabéticamente.
mapa_educacion = {
    7: "0. Sin instrucción",
    1: "1. Primaria Incompleta",
    2: "2. Primaria Completa",
    3: "3. Secundaria Incompleta",
    4: "4. Secundaria Completa",
    5: "5. Superior Incompleta",
    6: "6. Superior Completa"
}
datos["Nivel_Educativo"] = datos["NIVEL_ED_limpio"].map(mapa_educacion)

# ==========================================
# 4. CÁLCULO DE PORCENTAJES (Promedio Anual)
# ==========================================
datos_validos = datos.dropna(subset=['Nivel_Educativo'])

# Paso A: La "foto" trimestral sin duplicar
ed_trimestral = datos_validos.groupby(
    ["ANIO_ARCHIVO", "TRIMESTRE_ARCHIVO", "Nombre_Ciudad", "Nivel_Educativo"]
)["PONDERA"].sum().reset_index()

# Paso B: Promediamos los trimestres
ed_anual = ed_trimestral.groupby(
    ["ANIO_ARCHIVO", "Nombre_Ciudad", "Nivel_Educativo"]
)["PONDERA"].mean()

# Paso C: Calculamos los porcentajes
porcentajes_ed = (ed_anual / ed_anual.groupby(level=["ANIO_ARCHIVO", "Nombre_Ciudad"]).transform('sum')) * 100

# Paso D: Formato de tabla (redondeado a 1 decimal)
tabla_ed = porcentajes_ed.unstack(fill_value=0).round(1)

print("\n--- 2. Distribución Anual por Nivel Educativo ---")
print(tabla_ed)

# ==========================================
# 5. GENERACIÓN DEL GRÁFICO
# ==========================================
df_resistencia = tabla_ed.xs('Gran Resistencia', level='Nombre_Ciudad')
df_posadas = tabla_ed.xs('Posadas', level='Nombre_Ciudad')

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# Usamos una paleta secuencial de Seaborn especial para datos ordinales (tonos azules/verdes)
# El 'Blues' va de más claro a más oscuro automáticamente
colores = sns.color_palette("Blues", n_colors=7)

# Panel 1: Gran Resistencia
df_resistencia.plot(kind='bar', stacked=True, ax=axes[0], color=colores, width=0.85)
axes[0].set_title('Gran Resistencia', fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Año', fontsize=12)
axes[0].set_ylabel('Porcentaje de la Población (%)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend().set_visible(False)

# Panel 2: Posadas
df_posadas.plot(kind='bar', stacked=True, ax=axes[1], color=colores, width=0.85)
axes[1].set_title('Posadas', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('Año', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# Leyenda unificada a la derecha
axes[1].legend(title='Nivel Educativo Alcanzado', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# LO NUEVO: Etiquetas numéricas en las barras (solo mostramos si es mayor al 4% para no amontonar)
for ax in axes:
    for c in ax.containers:
        etiquetas = [f'{v.get_height():.1f}%' if v.get_height() > 4 else '' for v in c]
        # Usamos texto gris oscuro para que contraste bien tanto en fondos claros como oscuros
        ax.bar_label(c, labels=etiquetas, label_type='center', fontsize=8, fontweight='bold', color='#333333')

plt.suptitle('Evolución del Nivel Educativo Máximo Alcanzado (2016-2025)', fontsize=16, y=1.05)
plt.tight_layout()

plt.savefig("grafico_educacion_ordinal.png", dpi=300, bbox_inches="tight")
print("\nGráfico 'grafico_educacion_ordinal.png' generado y guardado con éxito.")
plt.show()