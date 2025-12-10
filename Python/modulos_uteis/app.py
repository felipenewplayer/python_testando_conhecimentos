import math
import datetime
import random
import time

# math.sqrt retorna a raiz quadrada de um número
print("Raiz quadrada de 16 é:", math.sqrt(16))

# math.sin retorna o seno de um ângulo em radianos
print("Seno de 45 é : ",math.sin(45))

# math.cos retorna o cosseno de um ângulo em radianos
print("Cosseno de 60 é : ",math.cos(60))

# math.log retorna o logaritmo natural de um número
print("Logaritmo natural de 10 é:", math.log(10))

print(math.log(64,2))

# math.e retorna número de Euler
print("Número de Euler é:", math.e)

# math.pi retorna o valor de pi
print("Valor de pi é:", math.pi)

# math.factorial retorna o fatorial de um número
print(math.factorial(5))



#### Datetime ####

print(dir(datetime))
print(datetime.date.today())
print(datetime.datetime.now().year)
print(datetime.datetime.now().month)
print(datetime.datetime.now().day)
print(datetime.datetime.now().hour)
print(datetime.datetime.now().minute)
print(datetime.datetime.now().second)


### Random ###
print(random.random())  # Gera um número aleatório entre 0 e 1 
print(random.randint(1, 10))  # Gera um número inteiro aleatório entre 1 e 10
print(random.choice(['maçã', 'banana', 'laranja']))  # Escolhe aleatoriamente um item de uma lista
print(random.sample(range(100), 5))  # Gera uma lista de 5

### Time ###

print(time.time())  # Retorna o tempo atual em segundos desde a época

print("Finalizando...")
time.sleep(2)  # Pausa a execução por 2 segundos
print("...")
time.sleep(2)
print("Programa finalizado.")