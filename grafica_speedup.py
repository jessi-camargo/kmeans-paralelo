import matplotlib.pyplot as plt

procesos = [1, 2, 4]
speedup = [1.0000, 1.7371, 2.6163]

plt.figure(figsize=(8, 5))

plt.plot(procesos, speedup, marker='o')

plt.title("Speedup de K-Means MPI")
plt.xlabel("Número de procesos MPI")
plt.ylabel("Speedup")

plt.xticks(procesos)
plt.grid(True)

plt.tight_layout()
plt.savefig("grafica_speedup.png", dpi=300)

plt.show()
