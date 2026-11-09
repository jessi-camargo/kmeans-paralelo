import numpy as np
from sklearn.datasets import make_blobs
import argparse
import os

def generar_datos(n_muestras):

    n_clusters = 4
    n_variables = 2

    # Generar datos de prueba
    X, etiquetas = make_blobs(
        n_samples=n_muestras,
        centers=n_clusters,
        n_features=n_variables,
        random_state=42
    )

    X = X.astype(np.float64)

    # Seleccionar centroides iniciales
    rng = np.random.default_rng(123)
    indices = rng.choice(
        len(X),
        size=n_clusters,
        replace=False
    )

    centroides = X[indices].copy()

    # Guardar los archivos
    os.makedirs("datos", exist_ok=True)

    np.save("datos/X.npy", X)
    np.save("datos/centroides_iniciales.npy", centroides)
    np.save("datos/etiquetas_referencia.npy", etiquetas)

    print("DATOS GENERADOS CORRECTAMENTE")
    print("--------------------------------")
    print("Cantidad de puntos:", len(X))
    print("Cantidad de variables:", n_variables)
    print("Cantidad de grupos:", n_clusters)
    print("Forma de los datos:", X.shape)
    print("Centroides iniciales:")
    print(centroides)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=100000,
        help="Cantidad de puntos"
    )

    args = parser.parse_args()

    generar_datos(args.n)
