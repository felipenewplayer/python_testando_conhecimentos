import numpy as np
from sklearn.linear_model import LogisticRegression

# Dados (entrada: idade)

idade = np.array([8,9,10,11,12,13,14,16,17,18,19,20,21,25,27,40,56]).reshape(-1,1)


# Saída (0 = menor de idade, 1 = maior de idade)

classe = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])

# Criando modelo de classificação

modelo = LogisticRegression()
modelo.fit(idade, classe)

print(modelo.predict([[16]])) 
print(modelo.predict([[21]])) 

print(modelo.predict_proba([[16]]))
print(modelo.predict_proba([[18]]))
print(modelo.predict_proba([[21]]))