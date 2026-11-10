import matplotlib.pyplot as plt

procesos = [1, 2, 4]
eficiencia = [100, 86.85, 65.41]

plt.figure(figsize=(8, 5))

plt.plot(procesos, eficiencia, marker='o')

plt.title("Eficiencia paralela de K-Means MPI")
plt.xlabel("Número de procesos MPI")
plt.ylabel("Eficiencia (%)")

plt.xticks(procesos)
plt.ylim(0, 110)
plt.grid(True)

plt.tight_layout()
plt.savefig("grafica_eficiencia.png", dpi=300)

plt.show()
