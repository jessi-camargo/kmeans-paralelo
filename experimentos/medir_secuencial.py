import numpy as np
import pandas as pd
import time

from secuencial.kmeans_secuencial import kmeans_secuencial

X = np.load("datos/X.npy")
centroides_iniciales = np.load(
    "datos/centroides_iniciales.npy"
)

repeticiones = 5
tiempos = []
resultados = []

print("EXPERIMENTO K-MEANS SECUENCIAL")
print("--------------------------------")
print("Puntos:", len(X))

# Calentamiento previo, fuera del tiempo medido
print("Ejecutando calentamiento...")
kmeans_secuencial(X, centroides_iniciales)

for repeticion in range(repeticiones):

    inicio = time.perf_counter()

    centroides, etiquetas, inercia, iteraciones = (
        kmeans_secuencial(
            X,
            centroides_iniciales
        )
    )

    tiempo = time.perf_counter() - inicio
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

media = np.mean(tiempos)
mediana = np.median(tiempos)
desviacion = np.std(tiempos, ddof=1)

print("\nRESULTADOS FINALES")
print("--------------------------------")
print("Media:", media)
print("Mediana:", mediana)
print("Desviación estándar:", desviacion)
print("Inercia:", inercia)
print("Iteraciones:", iteraciones)

df = pd.DataFrame(resultados)

df["Media"] = media
df["Mediana"] = mediana
df["Desviacion_Estandar"] = desviacion

nombre = (
    f"resultados/tablas/"
    f"tiempos_secuencial_{len(X)}.csv"
)

df.to_csv(nombre, index=False)

print("\nArchivo guardado:", nombre)

