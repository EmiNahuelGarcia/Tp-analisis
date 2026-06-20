# consultas.py
from carga_de_datos import cargar_datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar todos los datos
datos = cargar_datos()

# Ejemplo 1: Edad promedio en 2016 T2
anio, trimestre = 2016, 2
datos_trim = datos[(datos["ANIO_ARCHIVO"] == anio) & (datos["TRIMESTRE_ARCHIVO"] == trimestre)]
edad_promedio = datos_trim["CH06"].replace(-1, None).mean()
print(f"Edad promedio en {anio} T{trimestre}: {edad_promedio:.2f} años")

# Ejemplo 2: Distribución por sexo en 2025 T4
anio, trimestre = 2025, 4
datos_trim = datos[(datos["ANIO_ARCHIVO"] == anio) & (datos["TRIMESTRE_ARCHIVO"] == trimestre)]
sexo_dist = datos_trim["CH04"].map({1: "Varón", 2: "Mujer"}).value_counts()
print(f"\nDistribución por sexo en {anio} T{trimestre}:")
print(sexo_dist)


# Filtrar el trimestre y la ciudad
datos_trim = datos[
    (datos["ANIO_ARCHIVO"] == 2024) &
    (datos["TRIMESTRE_ARCHIVO"] == 4) &
    (datos["Nombre_Ciudad"] == "Gran Resistencia")
]

# Calcular activos (ocupados + desocupados) y desocupados ponderados
activos = datos_trim[datos_trim['ESTADO'].isin([1, 2])]['PONDERA'].sum()
desocupados = datos_trim[datos_trim['ESTADO'] == 2]['PONDERA'].sum()

# Calcular tasa de desocupación
tasa_desocupacion = (desocupados / activos * 100) if activos > 0 else 0

print(f"Tasa de desocupación en Gran Resistencia (2024 T4): {tasa_desocupacion:.2f}%")


# Distribución ponderada por sexo y ciudad en 2025 T4
anio, trimestre = 2025, 4
datos_trim = datos[(datos["ANIO_ARCHIVO"] == anio) & (datos["TRIMESTRE_ARCHIVO"] == trimestre)].copy()

datos_trim["Sexo"] = datos_trim["CH04"].map({1: "Varón", 2: "Mujer"})

sexo_ciudad_pond = datos_trim.groupby(["Nombre_Ciudad","Sexo"])["PONDERA"].sum().reset_index(name="Habitantes")

# Calcular porcentaje dentro de cada ciudad
sexo_ciudad_pond["Porcentaje"] = sexo_ciudad_pond.groupby("Nombre_Ciudad")["Habitantes"].transform(lambda x: x / x.sum() * 100)

print(sexo_ciudad_pond)

# Gráfico comparativo
plt.figure(figsize=(8,6))
sns.barplot(x="Nombre_Ciudad", y="Porcentaje", hue="Sexo", data=sexo_ciudad_pond, palette="Set2")
plt.title(f"Distribución porcentual por sexo y ciudad - {anio} T{trimestre}")
plt.ylabel("Porcentaje (%)")
plt.show()

