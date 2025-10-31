nome = "Felipe é íncrivel"

print(nome[1])


print(len(nome))
print(max(nome))
print(nome.upper())
print(nome.lower())
print(nome.capitalize())
print(nome.title())
print(nome.split())
print(nome.strip())
print(nome.replace("Felipe", "Thalya"))


texto = "Python é uma linguagem orientada a objetos"
print(len(texto))


print(f"A letra (a) no texto '{texto}' aparece{texto.count('a')} vezes.")
email = "felipe@gmail.com"
print("z" in email)


frase = "texto aleatório"
frase_invertida = " ".join(frase.split()[::-1])
print(frase_invertida)




nome = "Felipe"
nome_invertida = nome[::1]
print(nome_invertida)


frase_nova = "Jogar bola é daora"
frase_nova_invertida = " ".join(frase_nova.split()[::-1])
print(frase_nova_invertida)

print(frase_nova.split())
print(frase_nova.join())