import argparse
import time
import numpy as np

from numba import njit, prange, set_num_threads, get_num_threads

@njit(parallel=True)
def asignar_etiquetas(X, centroides):
    n = X.shape[0]
    k = centroides.shape[0]
    d = X.shape[1]

    etiquetas = np.empty(n, dtype=np.int32)

    for i in prange(n):
        mejor = 0
        menor = np.inf

        for c in range(k):
            distancia = 0.0

            for j in range(d):
                diferencia = X[i, j] - centroides[c, j]
                distancia += diferencia * diferencia

            if distancia < menor:
                menor = distancia
                mejor = c

        etiquetas[i] = mejor

    return etiquetas


@njit
def calcular_inercia(X, centroides):
    n = X.shape[0]
    k = centroides.shape[0]
    d = X.shape[1]

    total = 0.0

    for i in range(n):
        menor = np.inf

        for c in range(k):
            distancia = 0.0

            for j in range(d):
                diferencia = X[i, j] - centroides[c, j]
                distancia += diferencia * diferencia

            if distancia < menor:
                menor = distancia

        total += menor

    return total


@njit(parallel=True)
def acumular_centroides(X, etiquetas, k):
    n = X.shape[0]
    d = X.shape[1]
    tam_bloque = 4096
    n_bloques = (n + tam_bloque - 1) // tam_bloque

    sumas_bloques = np.zeros((n_bloques, k, d))
    conteos_bloques = np.zeros((n_bloques, k), dtype=np.int64)

    for b in prange(n_bloques):
        inicio = b * tam_bloque
        fin = min(inicio + tam_bloque, n)

        for i in range(inicio, fin):
            c = etiquetas[i]
            conteos_bloques[b, c] += 1

            for j in range(d):
                sumas_bloques[b, c, j] += X[i, j]

    sumas = np.zeros((k, d))
    conteos = np.zeros(k, dtype=np.int64)

    for b in range(n_bloques):
        for c in range(k):
            conteos[c] += conteos_bloques[b, c]

            for j in range(d):
                sumas[c, j] += sumas_bloques[b, c, j]

    return sumas, conteos




def kmeans_numba(X, centroides_iniciales,
                 max_iter=100, tolerancia=1e-4):

    centroides = centroides_iniciales.copy()

    for iteracion in range(max_iter):
        etiquetas = asignar_etiquetas(X, centroides)

        sumas, conteos = acumular_centroides(
            X, etiquetas, len(centroides)
        )

        nuevos = centroides.copy()

        for c in range(len(centroides)):
            if conteos[c] > 0:
                nuevos[c] = sumas[c] / conteos[c]

        diferencia = np.max(np.abs(nuevos - centroides))
        centroides = nuevos

        if diferencia <= tolerancia:
            break

    etiquetas = asignar_etiquetas(X, centroides)
    inercia = calcular_inercia(X, centroides)

    return centroides, etiquetas, inercia, iteracion + 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=100)

    args = parser.parse_args()

    if args.threads < 1:
        parser.error("Los hilos deben ser mayores que cero")

    set_num_threads(args.threads)

    X = np.load("datos/X.npy")
    iniciales = np.load("datos/centroides_iniciales.npy")

    print("K-MEANS PARALELO CON NUMBA")
    print("==========================")
    print("Puntos:", len(X))
    print("Variables:", X.shape[1])
    print("Hilos:", get_num_threads())

    # Compilación previa para no incluirla en el tiempo medido.
    muestra = X[:min(1000, len(X))]
    kmeans_numba(muestra, iniciales, args.max_iter)

    inicio = time.perf_counter()

    centroides, etiquetas, inercia, iteraciones = kmeans_numba(
        X, iniciales, args.max_iter
    )

    tiempo = time.perf_counter() - inicio

    np.save("resultados/centroides_numba.npy", centroides)
    np.save("resultados/etiquetas_numba.npy", etiquetas)

    print("\nRESULTADOS")
    print("==========================")
    print("Centroides:")
    print(centroides)
    print("Iteraciones:", iteraciones)
    print("Inercia:", inercia)
    print("Tiempo:", tiempo, "segundos")


if __name__ == "__main__":
    main()




