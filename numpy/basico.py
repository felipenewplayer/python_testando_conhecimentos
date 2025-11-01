import numpy as np

#Criando Arrays
numeros = np.array([10,20,30,40,50,60])
print(numeros)

#Operações Vetoriais(Sem Loop!)
print(numeros * 2)
print(numeros * 3)

#Estatísticas
print("Soma:", np.sum(numeros))
print("Média:", np.mean(numeros))
print("Maior:", np.max(numeros))
print("Menor:", np.min(numeros))

numeross = np.array([1,2,3,4,5,6,7,8,9,10])
dobrados = list(map(lambda n : n * 2, numeross))
print(dobrados)

filtrado = list(filter(lambda n : n > 5, numeross))
print(filtrado)

media = np.mean(numeross)
print(media)