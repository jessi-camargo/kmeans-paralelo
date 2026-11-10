
import numpy as np
import pandas as pd
import time

from numba import set_num_threads
import sys
sys.path.insert(0, ".")
from paralelo_numba.kmeans_numba import kmeans_numba

# Cargar datos
X = np.load("datos/X.npy")
centroides_iniciales = np.load(
    "datos/centroides_iniciales.npy"
)

# Configuraciones de hilos
hilos = [1, 2, 4, 8]
repeticiones = 5

resultados = []

print("EXPERIMENTO K-MEANS NUMBA")
print("=========================")

for numero_hilos in hilos:

    set_num_threads(numero_hilos)
    tiempos = []

    print(f"\nConfiguración: {numero_hilos} hilos")

    # Calentamiento: compilar antes de medir
    kmeans_numba(
        X[:1000],
        centroides_iniciales
    )

    for repeticion in range(repeticiones):

        inicio = time.perf_counter()

        centroides, etiquetas, inercia, iteraciones = (
            kmeans_numba(
                X,
                centroides_iniciales
            )
        )

        tiempo = time.perf_counter() - inicio
        tiempos.append(tiempo)

        print(
            f"Repetición {repeticion + 1}: "
            f"{tiempo:.6f} segundos"
        )

    media = np.mean(tiempos)
    desviacion = np.std(tiempos, ddof=1)

    resultados.append({
        "Hilos": numero_hilos,
        "Cantidad_Datos": len(X),
        "Tiempo_Medio": media,
        "Desviacion_Estandar": desviacion,
        "Inercia": inercia,
        "Iteraciones": iteraciones
    })

    print(f"Media: {media:.6f} segundos")

# Guardar resultados
df = pd.DataFrame(resultados)

df.to_csv(
    "resultados/tiempos_numba.csv",
    index=False
)

print("\nRESULTADOS FINALES")
print(df.to_string(index=False))

print("\nArchivo guardado: resultados/tiempos_numba.csv")
