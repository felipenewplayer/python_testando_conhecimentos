import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Criação dos meus dados com numpy
idade = np.random.randint(10,50,10)
altura = np.random.randint(140,200,10)

print(idade)
print(altura)

#Tratando e manipulando os dados com Pandas

df = pd.DataFrame({
    'Idade': idade,
    'Altura': altura
})

print(df.head())

#Cria de um gráfico com Matplolib
df.plot(kind='scatter', x='Idade', y='Altura')
plt.title('Relação entre altura e idade')
plt.ylabel('ALtura')
plt.xlabel('Idade')
plt.show()


