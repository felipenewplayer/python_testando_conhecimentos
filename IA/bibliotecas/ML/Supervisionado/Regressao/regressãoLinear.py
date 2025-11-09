import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model  import LinearRegression

altura = np.array([130,150,160,170,170,170]).reshape(-1,1) # matriz
idade = np.array([15,17,19,21,23,25])

df = pd.DataFrame({
    'Altura':altura.flatten(),
    'Idade':idade
})


print(df)

modelo = LinearRegression()
modelo.fit(altura,idade)
predicao = modelo.predict([[180]])
print(f'Idade previta {predicao}')

sns.barplot(x='Altura', y='Idade',data=df, color='purple')
plt.title("Relação entre Altura e Idade")
plt.show()

