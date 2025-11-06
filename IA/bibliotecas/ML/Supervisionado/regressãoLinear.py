import numpy as np
from sklearn.linear_model import LinearRegression


# Modelo simples que prevê a altura com base na idade
# Os dados são criado usando a lib Numpy 

idade= np.array([10,20,30,40,50,60,70]).reshape(-1,1) # Usando reshape para transforma esse vetor em matriz, porque queremos preves a altura, pois é o nosso dado de entrada (X)
altura = np.array([100,170,170,170,170,165,150])

# Criação do nosso modelo *--*

modelo = LinearRegression()
modelo.fit(idade, altura)

print(modelo.predict([[45]]))
print(modelo.coef_)      # inclinação da reta
print(modelo.intercept_) # onde a reta cruza o eixo Y


