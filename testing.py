import pandas as pd
from carga_de_datos import cargar_datos

# Cargar datos
datos = cargar_datos()

# Ver valores no nulos
pp04d = datos['PP04D_COD'].dropna()

print("\n=== Tipo de dato ===")
print(pp04d.dtype)

print("\n=== 50 ejemplos aleatorios ===")
print(pp04d.astype(str).sample(50, random_state=42).tolist())

print("\n=== Primeros 50 valores únicos ===")
print(sorted(pp04d.astype(str).unique())[:50])

print("\n=== Longitud de los códigos ===")
print(
    pp04d.astype(str)
         .str.split('.')
         .str[0]
         .str.len()
         .value_counts()
         .sort_index()
)

print("\n=== Frecuencia de los últimos dígitos ===")
print(
    pp04d.astype(str)
         .str.split('.')
         .str[0]
         .str[-1]
         .value_counts()
         .sort_index()
)

print("\n=== Frecuencia del tercer dígito ===")

def tercer_digito(x):
    x = str(x).split('.')[0]
    if len(x) >= 3:
        return x[2]
    return None

print(
    pp04d.apply(tercer_digito)
         .value_counts()
         .sort_index()
)