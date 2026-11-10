
import pandas as pd

secuencial = pd.read_csv(
    "resultados/tablas/tiempos_secuencial_100000.csv"
)

numba = pd.read_csv(
    "resultados/tiempos_numba.csv"
)

tiempo_secuencial = secuencial["Tiempo_Segundos"].mean()

tiempo_numba_1 = numba.loc[
    numba["Hilos"] == 1, "Tiempo_Medio"
].iloc[0]

comparacion = numba[
    ["Hilos", "Tiempo_Medio", "Desviacion_Estandar"]
].copy()

comparacion["Tiempo_Secuencial"] = tiempo_secuencial

# Aceleración respecto a Python secuencial
comparacion["Speedup_Secuencial"] = (
    tiempo_secuencial / comparacion["Tiempo_Medio"]
)

# Aceleración respecto a Numba con un hilo
comparacion["Speedup_Paralelo"] = (
    tiempo_numba_1 / comparacion["Tiempo_Medio"]
)

# Eficiencia paralela
comparacion["Eficiencia"] = (
    comparacion["Speedup_Paralelo"] /
    comparacion["Hilos"]
) * 100

print("\nCOMPARACION DE RENDIMIENTO")
print("==========================")
print(comparacion.to_string(index=False))

comparacion.to_csv(
    "resultados/tablas/comparacion_final.csv",
    index=False
)

print("\nArchivo guardado:")
print("resultados/tablas/comparacion_final.csv")

