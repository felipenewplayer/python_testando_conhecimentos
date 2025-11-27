from functools import reduce
pessoas = {
 "nome"   :"Felipe",
 "peso": 55.5,
 "idade" : 38
}

print(pessoas["nome"])
print(pessoas.get("idade")) # acessa com segurança
print(pessoas.keys()) # retorna as chaves
print(pessoas.values()) # retornar os valores
print(pessoas.items()) # retorna tuplas (chave, valor)

pessoas["profissão"] = "dev" # adicionar
pessoas["idade"] = 36 # atualizar
del pessoas["nome"]  # deletar
print(pessoas)

estoque = [
    {"nome": "Mouse", "preco": 80, "qtd": 10},
    {"nome": "Teclado", "preco": 120, "qtd": 5},
    {"nome": "Monitor", "preco": 900, "qtd": 2}
]

for item in estoque:
    total = item["qtd"] * item["preco"]
    print(f"{item['nome']} --- {total}")



valor_total = reduce(lambda acc,  item: acc + (item["preco"] * item["qtd"]), estoque, 0)
print(valor_total)



produtos = [
    {"nome": "Mouse", "preco": 80, "qtd": 10},
    {"nome": "Teclado", "preco": 120, "qtd": 5},
    {"nome": "Monitor", "preco": 900, "qtd": 2},
    {"nome": "Fone", "preco": 60, "qtd": 8},
    {"nome": "Cadeira Gamer", "preco": 1500, "qtd": 1},
]

caros = list(filter(lambda p : p["preco"] > 100, produtos))
print("Produtos acima de R$100: ")
for caro in caros:
    print(caro)



com_desconto = list(map(lambda p: {**p, "preco":round(p["preco"] * 0.9, 2)}, caros))
print("\nProdutos com desconto: ")
for p in com_desconto:
    print(p)
    
valor_totall = reduce(lambda acc, p: acc + p["preco"] * p["qtd"], produtos, 0)
print(f"Valor total dos produtos: R${valor_totall}")

pessoa = {"nome":"Felipe", "idade":25}
pessoa_nova = {**pessoa, "profissão":"DevOcultoDeJava"}
print(pessoa)
print(pessoa_nova)

numeros1 = {"a": 1,"b":2,"c3":3}
numeros2 = {"d":4,"e": 5,"f":6}
numeros3 = {**numeros1, **numeros2}
print(numeros3.values())
