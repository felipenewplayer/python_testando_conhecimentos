import numpy as np

### Matriz ###
matriz = np.array([
    (1,2,3),
    (4,5,6),
    (7,8,9)
])

print("Matriz")
print(matriz)

print("Soma de todas as células:", np.sum(matriz))
print("Média da matriz: ", np.mean(matriz))
print("Menor número da matriz: ", np.min(matriz))
print("Maior número da matriz" , np.max(matriz))

print("Soma por linha: ", np.sum(matriz, axis=1))
print("Soma por coluna: ", np.sum(matriz, axis=0))

c =np.zeros((4,3))
d = np.eye((10))
print(c)
print(d)
print(np.std(matriz))