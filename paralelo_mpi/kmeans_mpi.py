from mpi4py import MPI
import numpy as np
import time
import os


def kmeans_mpi(datos, k=3, max_iter=100, tolerancia=1e-4):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_total = None
    n_features = None

    if rank == 0:
        n_total, n_features = datos.shape

    n_total = comm.bcast(n_total, root=0)
    n_features = comm.bcast(n_features, root=0)

    # Dividir los datos entre los procesos
    cantidades = [n_total // size] * size

    for i in range(n_total % size):
        cantidades[i] += 1

    datos_locales = comm.scatter(
        np.array_split(datos, size) if rank == 0 else None,
        root=0
    )

    # Inicializar centroides en el proceso principal
    if rank == 0:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, k, replace=False)
        centroides = datos[indices].copy()
    else:
        centroides = None

    centroides = comm.bcast(centroides, root=0)

    for iteracion in range(max_iter):

        # Calcular distancias locales
        distancias = np.linalg.norm(
            datos_locales[:, np.newaxis, :] - centroides[np.newaxis, :, :],
            axis=2
        )

        etiquetas_locales = np.argmin(distancias, axis=1)

        # Sumar puntos por cluster
        sumas_locales = np.zeros((k, n_features))
        conteos_locales = np.zeros(k, dtype=int)

        for cluster in range(k):
            puntos = datos_locales[etiquetas_locales == cluster]

            if len(puntos) > 0:
                sumas_locales[cluster] = np.sum(puntos, axis=0)
                conteos_locales[cluster] = len(puntos)

        # Reducir resultados de todos los procesos
        sumas_globales = np.zeros((k, n_features))
        conteos_globales = np.zeros(k, dtype=int)

        comm.Allreduce(sumas_locales, sumas_globales, op=MPI.SUM)
        comm.Allreduce(conteos_locales, conteos_globales, op=MPI.SUM)

        # Actualizar centroides
        nuevos_centroides = centroides.copy()

        for cluster in range(k):
            if conteos_globales[cluster] > 0:
                nuevos_centroides[cluster] = (
                    sumas_globales[cluster] / conteos_globales[cluster]
                )

        # Verificar convergencia
        cambio = np.linalg.norm(nuevos_centroides - centroides)
        cambio_global = comm.allreduce(cambio, op=MPI.MAX)

        centroides = nuevos_centroides

        if cambio_global < tolerancia:
            break

    # Calcular inercia local
    distancias_finales = np.linalg.norm(
        datos_locales[:, np.newaxis, :] - centroides[np.newaxis, :, :],
        axis=2
    )

    etiquetas_locales = np.argmin(distancias_finales, axis=1)

    inercia_local = np.sum(
        np.min(distancias_finales, axis=1) ** 2
    )

    inercia_total = comm.reduce(inercia_local, op=MPI.SUM, root=0)

    # Reunir etiquetas en proceso principal
    etiquetas = comm.gather(etiquetas_locales, root=0)

    if rank == 0:
        etiquetas = np.concatenate(etiquetas)

        return centroides, etiquetas, inercia_total, iteracion + 1

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        ruta_datos = "datos/X.npy"

        if not os.path.exists(ruta_datos):
            print(f"No se encontró el archivo: {ruta_datos}")
            print("Genera los datos antes de ejecutar el programa.")
            datos = None
        else:
            datos = np.load(ruta_datos)
    else:
        datos = None

    inicio = time.perf_counter()

    resultado = kmeans_mpi(datos, k=3)

    fin = time.perf_counter()

    if rank == 0 and resultado is not None:
        centroides, etiquetas, inercia, iteraciones = resultado

        print("K-Means MPI finalizado")
        print(f"Procesos: {comm.Get_size()}")
        print(f"Tiempo: {fin - inicio:.4f} segundos")
        print(f"Inercia: {inercia:.4f}")
        print(f"Iteraciones: {iteraciones}")

        os.makedirs("resultados", exist_ok=True)

        np.save("resultados/centroides_mpi.npy", centroides)
        np.save("resultados/etiquetas_mpi.npy", etiquetas)
