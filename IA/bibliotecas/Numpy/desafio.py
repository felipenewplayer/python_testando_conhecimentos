import numpy as np
# 1
arr = np.array([1,2,3,4,5,6,7,8,9,10])
print(np.sum(arr))

# 2

arr2 = np.random.randint(0,100,50)

print(f'Média {arr2.mean()}')

# 3

arr2D = np.array([1,2]).reshape(-1,1)
print(arr2D)

# 4

arr_numeros = np.array([10,20,30,40])
arr_numeros2 = np.array([50,60,80,90])

print(arr_numeros + arr_numeros2)
print(arr_numeros * arr_numeros2)
print(arr_numeros2 - arr_numeros)
print(arr_numeros2 / arr_numeros)


idades = np.random.randint(5, 80 , 100)
print(idades)


meu_array = np.random.randint(0 , 10, 5)
media = meu_array.mean()
desvio_padrao = meu_array.std()
vacancia = meu_array.var()
print(f'Média do meu arr {media}')
print(f'Desvio padrão  do meu arr {desvio_padrao}')
print(f'Vacância do meu arr {vacancia}')



arr_numeros3 = np.array([1,2,3,4,5,6,7,8,9,10])
primeios_5 = arr_numeros3[0:5]
ultimos_5 = arr_numeros3[5:10]
print(primeios_5)
print(ultimos_5)


list_numeros = np.array([1,2,3,4,5,6,7,8,9,10])
print(list_numeros)

arr_pares = np.arange(2,22, 2)
print(f'Lista com números pares de 2 à 20 {arr_pares}')


arr_2D = np.array([[1,2,8],[4,7,9],[4,2,1]])
print(arr_2D)

formato_matriz = arr_2D.shape
print(formato_matriz)
numero_de_dimensoes = arr_2D.ndim
print(numero_de_dimensoes)
tipo_de_dados = arr_2D.dtype
print(tipo_de_dados)

print(arr_2D[2:3])
print(arr_2D[0])
print(arr_2D[:, 2])


arr_aleatorio = np.random.randint(1,10,(3,3))
print(f'\nMatriz aleatória \n{arr_aleatorio}')

soma = arr_aleatorio.sum()
media = arr_aleatorio.mean()
print(f'Soma geral {soma}')
print(f'Média geral {media}')
soma_coluna = arr_aleatorio.sum(axis=1)
soma_linha = arr_aleatorio.sum(axis=0)
print(soma_coluna)
print(soma_linha)

print(f'Maiores números das linhas {arr_aleatorio.max(axis=0)}')
import numpy as np

# 1
arr = np.array([1,2,3,4,5,6,7,8,9,10])
print(np.sum(arr))

# 2

arr2 = np.random.randint(0,100,50)

print(f'Média {arr2.mean()}')

# 3

arr2D = np.array([1,2]).reshape(-1,1)
print(arr2D)

# 4

arr_numeros = np.array([10,20,30,40])
arr_numeros2 = np.array([50,60,80,90])

print(arr_numeros + arr_numeros2)
print(arr_numeros * arr_numeros2)
print(arr_numeros2 - arr_numeros)
print(arr_numeros2 / arr_numeros)


idades = np.random.randint(5, 80 , 100)
print(idades)


meu_array = np.random.randint(0 , 10, 5)
media = meu_array.mean()
desvio_padrao = meu_array.std()
vacancia = meu_array.var()
print(f'Média do meu arr {media}')
print(f'Desvio padrão  do meu arr {desvio_padrao}')
print(f'Vacância do meu arr {vacancia}')



arr_numeros3 = np.array([1,2,3,4,5,6,7,8,9,10])
primeios_5 = arr_numeros3[0:5]
ultimos_5 = arr_numeros3[5:10]
print(primeios_5)
print(ultimos_5)


list_numeros = np.array([1,2,3,4,5,6,7,8,9,10])
print(list_numeros)

arr_pares = np.arange(2,22, 2)
print(f'Lista com números pares de 2 à 20 {arr_pares}')


arr_2D = np.array([[1,2,8],[4,7,9],[4,2,1]])
print(arr_2D)

formato_matriz = arr_2D.shape
print(formato_matriz)
numero_de_dimensoes = arr_2D.ndim
print(numero_de_dimensoes)
tipo_de_dados = arr_2D.dtype
print(tipo_de_dados)

print(arr_2D[2:3])
print(arr_2D[0])
print(arr_2D[:, 2])


arr_aleatorio = np.random.randint(1,10,(3,3))
print(f'\nMatriz aleatória \n{arr_aleatorio}')

soma = arr_aleatorio.sum()
media = arr_aleatorio.mean()
print(f'Soma geral {soma}')
print(f'Média geral {media}')
soma_coluna = arr_aleatorio.sum(axis=0)
soma_linha = arr_aleatorio.sum(axis=1)
print(f'Soma de cada coluna {soma_coluna}')
print(f'Soma de cada linha {soma_linha}')

print(f'Maiores números das linhas {arr_aleatorio.max(axis=1)}')
print(f'Maiores números das colunas {arr_aleatorio.max(axis=0)}')


print(f'Menores números das linhas {arr_aleatorio.min(axis=1)}')
print(f'Menores números das colunas {arr_aleatorio.min(axis=0)}')

print(arr_aleatorio.reshape(-1,1))
print(f'Ordenar colunas \n{np.sort(arr_aleatorio, axis=0)}')
print(f'Ordenar linhas \n{np.sort(arr_aleatorio, axis=1)}')
print(f'Mostrar o maior índice de cada linha \n{np.argmax(arr_aleatorio, axis=1)}')
print(f'Mostrar o maior índice de cada coluna \n{np.argmax(arr_aleatorio, axis=0)}')


# duas matriz

matriz1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
matriz2 = np.array([[9,8,7],[6,5,4],[3,2,1]])

print(f'\nMultiplicaçãio de matriz \n{matriz1 * matriz2}')
print(f'\Soma de matriz \n{matriz1 + matriz2}')
print(f'\Subtração de matriz \n{matriz1 - matriz2}')
print(f'\Divisão de matriz \n{matriz1 / matriz2}')

print(np.dot(matriz1,matriz2))

