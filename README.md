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
