
import pandas as pd
import matplotlib.pyplot as plt
import os

# Crear carpeta para las gráficas
os.makedirs("resultados/graficas", exist_ok=True)

# Leer resultados
secuencial = pd.read_csv(
    "resultados/tablas/tiempos_secuencial_100000.csv"
)

numba = pd.read_csv(
    "resultados/tiempos_numba.csv"
)

comparacion = pd.read_csv(
    "resultados/tablas/comparacion_final.csv"
)

# Tiempo secuencial promedio
tiempo_secuencial = secuencial["Tiempo_Segundos"].mean()

# GRÁFICA 1: Comparación de tiempos
etiquetas = ["Secuencial", "Numba 1 hilo",
             "Numba 2 hilos", "Numba 4 hilos",
             "Numba 8 hilos"]

tiempos = [
    tiempo_secuencial,
    *numba["Tiempo_Medio"].tolist()
]

plt.figure(figsize=(10, 6))

plt.bar(etiquetas, tiempos, color=[
    "gray", "blue", "green", "orange", "red"
])

plt.title("Comparación de tiempos de ejecución - K-Means")
plt.xlabel("Implementación")
plt.ylabel("Tiempo (segundos)")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("resultados/graficas/comparacion_tiempos.png")
plt.close()

# GRÁFICA 2: Speedup
plt.figure(figsize=(9, 6))

plt.plot(
    comparacion["Hilos"],
    comparacion["Speedup_Paralelo"],
    marker="o",
    linewidth=2,
    color="blue"
)

plt.title("Speedup de K-Means con Numba")
plt.xlabel("Número de hilos")
plt.ylabel("Speedup")
plt.xticks([1, 2, 4, 8])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultados/graficas/speedup.png")
plt.close()

# GRÁFICA 3: Eficiencia paralela
plt.figure(figsize=(9, 6))

plt.plot(
    comparacion["Hilos"],
    comparacion["Eficiencia"],
    marker="o",
    linewidth=2,
    color="green"
)

plt.title("Eficiencia paralela de K-Means")
plt.xlabel("Número de hilos")
plt.ylabel("Eficiencia (%)")
plt.xticks([1, 2, 4, 8])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultados/graficas/eficiencia.png")
plt.close()

print("GRAFICAS GENERADAS CORRECTAMENTE")
print("--------------------------------")
print("1. comparacion_tiempos.png")
print("2. speedup.png")
print("3. eficiencia.png")
