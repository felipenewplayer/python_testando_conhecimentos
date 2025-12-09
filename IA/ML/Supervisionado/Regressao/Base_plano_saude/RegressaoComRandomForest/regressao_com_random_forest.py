import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

base_plano_saude = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\plano_saude2.csv')

print(base_plano_saude)
X_plano_saude2 = base_plano_saude.iloc[:, 0 ].values.reshape(-1, 1)
y_plano_saude2 = base_plano_saude.iloc[:, 1 ].values

print(X_plano_saude2)


modelo_random_forest = RandomForestRegressor(n_estimators=10)
print(modelo_random_forest)
modelo_random_forest.fit(X_plano_saude2, y_plano_saude2)
print(modelo_random_forest.score(X_plano_saude2, y_plano_saude2))


X_teste = np.arange(min(X_plano_saude2), max(X_plano_saude2), 0.1)
print(X_teste.shape)
X_teste = X_teste.reshape(-1, 1)
print(X_teste.shape)


grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2, color_discrete_sequence=['blue'], title='Plano de Saúde x Idade')
grafico.add_scatter(x=X_teste.ravel(), y=modelo_random_forest.predict(X_teste), mode='lines', line=dict(color='red'), name='Regressão com RandomForest')
grafico.show()


print(modelo_random_forest.predict([[40]]))

