import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# Dados de entrada (peso, idade)
X = np.array([[50,1.60],[55, 1.65],[60,1.70],[70, 1.75],[85, 1.85],[95,1.75],[110, 1.75]])

# Classe: 0 = magro , 1 = normal, 2 = sobrepeso, 3 = obeso
y  = np.array([0,0,1,1,2,2,3])

# Criar modelo KNN
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X,y)

# Testa com alguém 
print(modelo.predict([[70, 1.80]]))

