from functools import reduce

numeros = [10,50,60,11,111,1240, 5632, 11,1,22,3,5,66,8,1]
numeros.sort()
print(numeros)

dobrados = list(
    filter(lambda n : n > 100 ,map(lambda n : n * 2, numeros)))

print(dobrados)

soma = reduce(lambda acumulador, n : acumulador + n, numeros)
print(soma)
    
pares = list(filter(lambda n : n % 2 == 0, numeros))
print(pares)

multiplacar = reduce(lambda acc, n : acc * n, pares)
print(multiplacar)

maior_numero = max(numeros)
menor_numero = min(numeros)
print(f"Maior número da lista {maior_numero}")
print(f"Menor número da lista { menor_numero}")


nomes = ["Felipe", "Rafa", "Jeniffer", "Robson", "Ana"]
for nome in nomes:
    if nome.startswith("A"):
        print(nome)

nomes.append("Joana")
nomes.append("Marcos")
nomes.append("Lucas")
print(nomes)

nome_digitado = input("Digite um nome para excluir da lista: ")
if nome_digitado in nomes:
    nomes.remove(nome_digitado)
    print(nomes)
else:
    print("Não tem esse nome!")
    
def media_das_notas(n1, n2, n3):
    notas = [n1, n2, n3]
    soma = sum(notas)
    media = soma / len(notas)
    
    if media >= 7 :
        print("Parabéns, aprovado!")
    else:
        print("Reprovado!")
media_das_notas(10,10,0)

for i in range(1,11):
    print(i * 1)
    

novos_numeros = [10,25,30,41,50,60,77,80,90,100]
filtrado_numeros = list(filter(lambda n : n > 50, novos_numeros))
print(filtrado_numeros)
pares_novos = list(filter(lambda n : n * 2 , novos_numeros))
print(pares_novos)
soma_dos_novos_numeros = reduce(lambda acc, n : acc + n, novos_numeros)
print(soma_dos_novos_numeros)

nomes_maiusculos = list(map(lambda n : n.upper(), nomes))
print(nomes_maiusculos)

nomes_menos_5_letras = list(filter(lambda n : len(n) < 5 , nomes))
print(nomes_menos_5_letras)



alunos = {
    "Felipe": [10,10,10],
    "Thayla": [10,1,1],
    "Tadei": [5,7,8]
}

