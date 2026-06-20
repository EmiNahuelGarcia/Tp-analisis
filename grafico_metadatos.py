import matplotlib.pyplot as plt
import numpy as np

# 1. Todas las categorías de tu tabla
# Abreviamos un poco los nombres para que entren bien en el eje X
categorias = [
    'CH04\nFalt.', 'CH04\nNo R.', 'CH04\nAtíp.',
    'ESTADO\nFalt.', 'ESTADO\nNo R.', 'ESTADO\nAtíp.',
    'NIVEL ED\nFalt.', 'NIVEL ED\nNo R.', 'NIVEL ED\nAtíp.',
    'CH06\nFalt.', 'CH06\nNo R.', 'CH06\nAtíp.',
    'P21\nFalt.', 'P21\nNo R.', 'P21\nAtíp.'
]

# 2. Datos exactos (Reemplazando los "-" por 0 para graficar)
posadas = [0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 525, 4231, 566]
gran_resistencia = [0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 1, 142, 2439, 586]

# 3. Configuración del gráfico (Más ancho para que entren las 15 barras)
x = np.arange(len(categorias))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))

# Dibujamos las barras
rects1 = ax.bar(x - width/2, posadas, width, label='Posadas', color='#e76f51')
rects2 = ax.bar(x + width/2, gran_resistencia, width, label='Gran Resistencia', color='#2a9d8f')

# 4. Textos y formato
ax.set_ylabel('Cantidad de casos', fontsize=11)
ax.set_xlabel('Indicador / Variable', fontsize=11)
ax.set_title('Detección de Valores Faltantes, No Respuesta y Atípicos (2016-2025)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categorias, fontsize=9)
ax.legend(title="Aglomerado")

# 5. Etiquetas numéricas arriba de cada barra
ax.bar_label(rects1, padding=3, fontsize=9)
ax.bar_label(rects2, padding=3, fontsize=9)

# Evita que se corten los bordes
fig.tight_layout()
plt.savefig('grafico_metadatos_completo.png', dpi=300)
print("Gráfico 'grafico_metadatos_completo.png' generado con éxito.")

plt.show()