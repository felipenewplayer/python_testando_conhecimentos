import numpy as np

arr = np.array([10,20,40,50]) # isso é um array (1D)
matriz = np.array([[1,2,3],[4,5,6]]) # isso é uma matriz(2D)

print(arr)
print(matriz)

soma_matriz = matriz.sum(axis=0)
print(soma_matriz)