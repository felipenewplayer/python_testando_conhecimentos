import numpy as np

#Sequência simples (igual o range)
seq = np.arange(0,10,2)
print("Sequência: ", seq)

# Array de zeros
zeros = np.zeros((2,2)) 
print("Zeros:" , zeros)

#Arrays de uns
uns = np.ones((2,4))
print("Uns: ", uns)

# Números aleátorios 
aleatorios = np.random.randint(1,1000, size=(10,10))
print("Números aleatórios: " , aleatorios)



dados = np.array([1,2,3,4,5,6])

print("Quadrados: ", np.square(dados))
print("Raiz quadrada: ", np.sqrt(dados))
print("Seno : ", np.sin(dados))
print("Logaritmo:" , np.log(dados))


matriz = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("Elemento [0,1]:", matriz[0, 1])      # Linha 0, Coluna 1
print("Linha 1:", matriz[1])                # Linha completa
print("Coluna 2:", matriz[:, 2])            # Todas as linhas, Coluna 2
print("Submatriz 2x2:\n", matriz[:2, :2])   # Linhas 0-1 e Colunas 0-1

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Soma:\n", A + B)
print("Subtração:\n", A - B)
print("Multiplicação elemento a elemento:\n", A * B)
print("Produto de matrizes (dot):\n", np.dot(A, B))