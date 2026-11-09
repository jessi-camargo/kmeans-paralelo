import numpy as np
import pandas as pd
import time

from secuencial.kmeans_secuencial import kmeans_secuencial


# Cargar datos
X = np.load("datos/X.npy")

centroides_iniciales = np.load(
    "datos/centroides_iniciales.npy"
)

# Número de repeticiones
repeticiones = 3

tiempos = []
resultados = []

print("EXPERIMENTO K-MEANS SECUENCIAL")
print("--------------------------------")

for repeticion in range(repeticiones):

    inicio = time.perf_counter()

    centroides, etiquetas, inercia, iteraciones = (
        kmeans_secuencial(
            X,
            centroides_iniciales
        )
    )

    fin = time.perf_counter()

    tiempo = fin - inicio

    tiempos.append(tiempo)

    resultados.append({
        "Repeticion": repeticion + 1,
        "Cantidad_Datos": len(X),
        "Tiempo_Segundos": tiempo,
        "Inercia": inercia,
        "Iteraciones": iteraciones
    })

    print(
        f"Repetición {repeticion + 1}: "
        f"{tiempo:.6f} segundos"
    )


# Calcular estadísticas
media = np.mean(tiempos)
desviacion = np.std(tiempos, ddof=1)

print("\nRESULTADOS FINALES")
print("--------------------------------")

print("Media:", media)
print("Desviación estándar:", desviacion)

# Guardar resultados
df = pd.DataFrame(resultados)

df["Media"] = media
df["Desviacion_Estandar"] = desviacion

nombre = (
    f"resultados/tablas/"
    f"tiempos_secuencial_{len(X)}.csv"
)

df.to_csv(nombre, index=False)

print("\nArchivo guardado:", nombre)
