import matplotlib.pyplot as plt

procesos = [1, 2, 4]
tiempos = [0.4228, 0.2434, 0.1616]

plt.figure(figsize=(8, 5))

plt.bar(procesos, tiempos)

plt.title("Comparación de tiempos de ejecución K-Means MPI")
plt.xlabel("Número de procesos MPI")
plt.ylabel("Tiempo (segundos)")
plt.xticks(procesos)

plt.tight_layout()
plt.savefig("grafica_tiempos.png", dpi=300)

plt.show()
