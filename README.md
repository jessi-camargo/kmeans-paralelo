# K-Means Paralelo desde Cero

## Descripción

Este proyecto implementa el algoritmo K-Means desde cero
para agrupar grandes cantidades de datos y comparar el
rendimiento entre diferentes modelos de procesamiento.

## Objetivo

Implementar y evaluar una versión secuencial y dos
versiones paralelas del algoritmo K-Means.

## Integrantes

- Persona 1: Implementación secuencial.
- Persona 2: Implementación paralela con Numba.
- Persona 3: Implementación distribuida con MPI.

## Herramientas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (generación de datos)
- Numba
- mpi4py
- OpenMPI

## Estructura

- secuencial/
- paralelo_numba/
- paralelo_mpi/
- experimentos/
- resultados/
- datos/

## Instalación

Crear entorno virtual:

python3 -m venv venv

Activar:

source venv/bin/activate

Instalar dependencias:

pip install -r requirements.txt

## Generar datos

python datos/generar_datos.py --n 100000

## Ejecutar versión secuencial

python secuencial/kmeans_secuencial.py

## Ejecutar experimentos secuenciales

python -m experimentos.medir_secuencial

## Implementación paralela con Numba

Se desarrolló una versión paralela de K-Means utilizando
Numba y su compilación JIT.

Se utilizaron las siguientes técnicas:

- Compilación de funciones mediante @njit.
- Paralelización de ciclos mediante prange.
- Acumulación de centroides mediante bloques.
- Configuración del número de hilos.

## Ejecutar K-Means con Numba

Ejecutar con un hilo:

python paralelo_numba/kmeans_numba.py --threads 1

Ejecutar con dos hilos:

python paralelo_numba/kmeans_numba.py --threads 2

Ejecutar con cuatro hilos:

python paralelo_numba/kmeans_numba.py --threads 4

Ejecutar con ocho hilos:

python paralelo_numba/kmeans_numba.py --threads 8

## Experimentos de rendimiento

Ejecutar:

python -m experimentos.medir_numba

python -m experimentos.medir_secuencial

python -m experimentos.comparar_resultados

python -m experimentos.graficas

## Resultados

Se realizaron pruebas con 100000 puntos.

La configuración de cuatro hilos obtuvo el mejor
rendimiento entre las configuraciones evaluadas.

| Hilos | Tiempo medio (s) |
|---|---:|
| 1 | 0.009044 |
| 2 | 0.005526 |
| 4 | 0.003418 |
| 8 | 0.004547 |

El mejor rendimiento se obtuvo utilizando cuatro hilos,
con un speedup de aproximadamente 2.65 respecto a Numba
con un hilo.

## Conclusiones

La implementación con Numba permitió mejorar
considerablemente el rendimiento respecto a la
implementación secuencial en Python.

El uso de cuatro hilos obtuvo el mejor resultado
en las pruebas realizadas.

El incremento del número de hilos no garantiza
una mejora proporcional del rendimiento.

## Implementación distribuida con MPI

Se desarrolló una versión distribuida del algoritmo K-Means utilizando MPI mediante la biblioteca mpi4py.

La implementación utiliza varios procesos para distribuir el conjunto de datos y realizar los cálculos de manera simultánea.

Se utilizaron las siguientes técnicas:

- Distribución de datos mediante `scatter`.
- Comunicación entre procesos mediante `broadcast`.
- Reducción de resultados mediante `Allreduce`.
- Recolección de etiquetas mediante `gather`.
- Actualización distribuida de centroides.
- Verificación de convergencia.

### Ejecutar K-Means con MPI

Ejecutar con un proceso:

```bash
mpirun -np 1 python paralelo_mpi/kmeans_mpi.py
```

Ejecutar con dos procesos:

```bash
mpirun -np 2 python paralelo_mpi/kmeans_mpi.py
```

Ejecutar con cuatro procesos:

```bash
mpirun -np 4 python paralelo_mpi/kmeans_mpi.py
```

### Resultados de rendimiento MPI

Se realizaron pruebas utilizando 1, 2 y 4 procesos MPI.

| Procesos | Tiempo (s) | Speedup | Eficiencia |
|---:|---:|---:|---:|
| 1 | 0.4228 | 1.0000 | 100 % |
| 2 | 0.2434 | 1.7371 | 86.85 % |
| 4 | 0.1616 | 2.6163 | 65.41 % |

En todas las ejecuciones se obtuvo una inercia de 736821.5648 y 29 iteraciones.

### Gráficas de rendimiento MPI

Se generaron las siguientes gráficas para analizar el comportamiento de la implementación:

- `grafica_tiempos.png`: comparación de tiempos de ejecución.
- `grafica_speedup.png`: aceleración obtenida al incrementar los procesos.
- `grafica_eficiencia.png`: eficiencia paralela de la implementación.

### Conclusión de MPI

Los resultados muestran que al incrementar el número de procesos MPI se reduce el tiempo de ejecución. La configuración de cuatro procesos obtuvo el menor tiempo, mientras que la configuración de dos procesos presentó una mayor eficiencia paralela.

El incremento de procesos no garantiza una mejora proporcional del rendimiento, debido a los costos de comunicación y sincronización entre procesos.
