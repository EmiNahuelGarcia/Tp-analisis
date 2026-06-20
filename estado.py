import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# 4. CÁLCULO DE PORCENTAJES (Promedio Anual)
# ==========================================
# Filtramos los nulos para hacer el cálculo matemáticamente exacto
datos_validos = datos.dropna(subset=['Condicion_Actividad'])

# Paso A: La "foto" trimestral sin duplicar
actividad_trimestral = datos_validos.groupby(
    ["ANIO_ARCHIVO", "TRIMESTRE_ARCHIVO", "Nombre_Ciudad", "Condicion_Actividad"]
)["PONDERA"].sum().reset_index()

# Paso B: Promediamos los trimestres (el dato anual real)
actividad_anual = actividad_trimestral.groupby(
    ["ANIO_ARCHIVO", "Nombre_Ciudad", "Condicion_Actividad"]
)["PONDERA"].mean()

# Paso C: Calculamos los porcentajes sobre ese promedio anual
porcentajes_actividad = (actividad_anual / actividad_anual.groupby(level=["ANIO_ARCHIVO", "Nombre_Ciudad"]).transform('sum')) * 100

# Paso D: Le damos formato de tabla y redondeamos a 1 decimal
tabla_final = porcentajes_actividad.unstack().round(1)

print("\n--- 2. Porcentaje Promedio Anual por Condición de Actividad y Ciudad ---")
print(tabla_final)




# ==========================================
# 5. GENERACIÓN DEL GRÁFICO (BARRAS APILADAS)
# ==========================================

# 1. Separamos la tabla principal en dos mini-tablas
df_resistencia = tabla_final.xs('Gran Resistencia', level='Nombre_Ciudad')
df_posadas = tabla_final.xs('Posadas', level='Nombre_Ciudad')

# 2. Configuramos el lienzo
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 3. Paleta de colores
colores = ['#d62828', '#f77f00', '#fcbf49', '#003049']

# 4. Graficamos Gran Resistencia
df_resistencia.plot(kind='bar', stacked=True, ax=axes[0], color=colores, width=0.85)
axes[0].set_title('Gran Resistencia', fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Año', fontsize=12)
axes[0].set_ylabel('Porcentaje de la Población Total (%)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend().set_visible(False)

# 5. Graficamos Posadas
df_posadas.plot(kind='bar', stacked=True, ax=axes[1], color=colores, width=0.85)
axes[1].set_title('Posadas', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('Año', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Condición de Actividad', bbox_to_anchor=(1.05, 1), loc='upper left')

# ====================================================================
# --- LO NUEVO: AGREGAR LOS NÚMEROS A LAS BARRAS ---
# Iteramos sobre ambos paneles (axes) y sobre cada bloque de color (containers)
for ax in axes:
    for c in ax.containers:
        # Filtramos: Si el valor es mayor a 3%, mostramos el texto con el %. Si no, lo dejamos vacío ('')
        etiquetas = [f'{v.get_height():.1f}%' if v.get_height() > 3 else '' for v in c]
        
        # Insertamos el texto en el centro (label_type='center') del bloque
        ax.bar_label(c, labels=etiquetas, label_type='center', fontsize=9, fontweight='bold', color='white')
# ====================================================================

# 7. Título principal y ajuste
plt.suptitle('Composición Demográfica por Condición de Actividad (2016-2025)', fontsize=16, y=1.05)
plt.tight_layout()

# 8. Guardamos la imagen
plt.savefig("grafico_estado_demografico_etiquetas.png", dpi=300, bbox_inches="tight")
print("\nGráfico 'grafico_estado_demografico_etiquetas.png' generado y guardado con éxito.")

plt.show()