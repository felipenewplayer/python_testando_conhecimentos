import random
listaDeNumeros = [10,2,1,3,5,6,8,20,44,66,7,88,4342]

tamanhoDaLista = len(listaDeNumeros)
print(f"Tamanho da lista {tamanhoDaLista}")

maiorNumeroDaLista = max(listaDeNumeros)
print(f"Maior número lista {maiorNumeroDaLista}")

menorNumeroDaLista = min(listaDeNumeros)
print(f"Menor número lista {menorNumeroDaLista}")

somaDosNumerosDaLista = sum(listaDeNumeros)
print(f"Soma dos números da lista {somaDosNumerosDaLista}")

listaDeNumeros.sort()
print(listaDeNumeros)

listaDeNumeros.reverse()
print(listaDeNumeros)

pares = [i for i in listaDeNumeros if i  % 2 == 0]
print(f"Números pares da lista {pares}")


nomes = ["Ana", "Rafa", "Alex"]
for nome in nomes:
    if nome.startswith("A"):
        print(f"{nome}")
        
        
lista_de_frutas = []
def adicionar_frutas(lista_de_frutas):
    nome_fruta = input("Digite o nome de uma fruta! ")
    lista_de_frutas.append(nome_fruta)
adicionar_frutas(lista_de_frutas)

def excluir_fruta(lista_de_frutas):
    print(lista_de_frutas)
    fruta_excluir = input("Qual fruta você quer excluir: ")
    if fruta_excluir in lista_de_frutas:
     lista_de_frutas.remove(fruta_excluir)
     print(f"Fruta excluída com sucesso!")
    else:
     print(f"{fruta_excluir} não está na lista.")
   
excluir_fruta(lista_de_frutas)



cidades = ["São Paulo", "Rio De Janeiro", "Rio Grande do Sul", "Acre"]
escolhido = random.choice(cidades)
print(escolhido)


        
        
        



