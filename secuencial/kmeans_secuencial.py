import numpy as np
import time


def calcular_distancias(X, centroides):

    distancias = np.sum(
        (X[:, np.newaxis, :] - centroides[np.newaxis, :, :]) ** 2,
        axis=2
    )

    return distancias


def asignar_grupos(X, centroides, tamano_lote=100000):

    n_muestras = len(X)
    n_clusters = len(centroides)

    etiquetas = np.empty(n_muestras, dtype=np.int32)

    sumas = np.zeros_like(centroides)
    cantidades = np.zeros(n_clusters, dtype=np.int64)

    # Procesar los datos por bloques
    for inicio in range(0, n_muestras, tamano_lote):

        fin = min(inicio + tamano_lote, n_muestras)

        bloque = X[inicio:fin]

        # Calcular distancias
        distancias = calcular_distancias(
            bloque,
            centroides
        )

        # Asignar cada punto al centroide más cercano
        etiquetas_bloque = np.argmin(
            distancias,
            axis=1
        )

        etiquetas[inicio:fin] = etiquetas_bloque

        # Calcular sumas parciales
        cantidades += np.bincount(
            etiquetas_bloque,
            minlength=n_clusters
        )

        for variable in range(X.shape[1]):

            sumas[:, variable] += np.bincount(
                etiquetas_bloque,
                weights=bloque[:, variable],
                minlength=n_clusters
            )

    return etiquetas, sumas, cantidades


def calcular_inercia(X, centroides, tamano_lote=100000):

    inercia = 0.0

    for inicio in range(0, len(X), tamano_lote):

        fin = min(inicio + tamano_lote, len(X))

        bloque = X[inicio:fin]

        distancias = calcular_distancias(
            bloque,
            centroides
        )

        distancias_minimas = np.min(
            distancias,
            axis=1
        )

        inercia += np.sum(distancias_minimas)

    return inercia


def kmeans_secuencial(
    X,
    centroides_iniciales,
    max_iter=100,
    tolerancia=1e-4
):

    centroides = centroides_iniciales.copy()

    for iteracion in range(max_iter):

        # Asignar puntos y obtener sumas parciales
        etiquetas, sumas, cantidades = asignar_grupos(
            X,
            centroides
        )

        nuevos_centroides = centroides.copy()

        # Actualizar centroides
        for k in range(len(centroides)):

            if cantidades[k] > 0:

                nuevos_centroides[k] = (
                    sumas[k] / cantidades[k]
                )

        # Comprobar convergencia
        diferencia = np.max(
            np.abs(nuevos_centroides - centroides)
        )

        centroides = nuevos_centroides

        if diferencia <= tolerancia:
            break

    # Calcular las asignaciones finales
    etiquetas, _, _ = asignar_grupos(
        X,
        centroides
    )

    # Calcular inercia final
    inercia = calcular_inercia(
        X,
        centroides
    )

    return centroides, etiquetas, inercia, iteracion + 1


if __name__ == "__main__":

    # Cargar los datos
    X = np.load("datos/X.npy")

    centroides_iniciales = np.load(
        "datos/centroides_iniciales.npy"
    )

    print("INICIANDO K-MEANS SECUENCIAL")
    print("--------------------------------")

    print("Cantidad de puntos:", len(X))
    print("Cantidad de variables:", X.shape[1])
    print("Cantidad de centroides:", len(centroides_iniciales))

    # Medir tiempo de ejecución
    inicio = time.perf_counter()

    centroides, etiquetas, inercia, iteraciones = (
        kmeans_secuencial(
            X,
            centroides_iniciales
        )
    )

    fin = time.perf_counter()

    tiempo = fin - inicio

    print("\nRESULTADOS")
    print("--------------------------------")

    print("Centroides finales:")
    print(centroides)

    print("\nCantidad de grupos:")

    for k in range(len(centroides)):

        cantidad = np.sum(etiquetas == k)

        print(f"Grupo {k}: {cantidad} puntos")

    print("\nIteraciones:", iteraciones)
    print("Inercia:", inercia)
    print("Tiempo de ejecución:", tiempo, "segundos")

    # Guardar resultados
    np.save(
        "resultados/centroides_secuencial.npy",
        centroides
    )

    np.save(
        "resultados/etiquetas_secuencial.npy",
        etiquetas
    )

    print("\nResultados guardados correctamente.")
