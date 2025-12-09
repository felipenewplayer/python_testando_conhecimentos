import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px

base_de_dados_saude = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\plano_saude2.csv')

print(base_de_dados_saude)
X_saude2 = base_de_dados_saude.iloc[:,0].values.reshape(-1, 1)
y_saude2 = base_de_dados_saude.iloc[:,1].values
print(X_saude2)
print(y_saude2)
regressor_arvore_decisao = DecisionTreeRegressor()
regressor_arvore_decisao.fit(X_saude2, y_saude2)

print(regressor_arvore_decisao.score(X_saude2, y_saude2))

grafico = px.scatter(x=X_saude2.flatten(), y=y_saude2, color_discrete_sequence=['blue'], title='Plano de Saúde x Idade')
grafico.add_scatter(x=X_saude2.flatten(), y=regressor_arvore_decisao.predict(X_saude2), mode='lines', line=dict(color='red'), name='Regressão com Árvore de Decisão')
grafico.show()

X_teste = np.arange(min(X_saude2), max(X_saude2), 0.1)
print(X_teste.shape)
X_teste = X_teste.reshape(-1, 1)
print(X_teste.shape)


grafico = px.scatter(x=X_saude2.ravel(), y=y_saude2, color_discrete_sequence=['blue'], title='Plano de Saúde x Idade')
grafico.add_scatter(x=X_teste.ravel(), y=regressor_arvore_decisao.predict(X_teste), mode='lines', line=dict(color='red'), name='Regressão com Árvore de Decisão')
grafico.show()


print(regressor_arvore_decisao.predict([[40]]))