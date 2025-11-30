import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

base_plano_saude = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Regressao\Base_plano_saude\plano_saude.csv')

print(base_plano_saude.head())

X_plano_saude = base_plano_saude.iloc[:,0].values
print(X_plano_saude)
y_plano_saude = base_plano_saude.iloc[:,1].values
print(y_plano_saude)

coeficiente = np.corrcoef(X_plano_saude, y_plano_saude)
print(coeficiente)

X_plano_saude = X_plano_saude.reshape(-1,1)
print(X_plano_saude)
print(X_plano_saude.shape)

regressao_linear = LinearRegression()
regressao_linear.fit(X_plano_saude, y_plano_saude)
print((regressao_linear.intercept_))
print((regressao_linear.coef_))

previsoes = regressao_linear.predict(X_plano_saude)
print(previsoes)

X_plano_saude.ravel()
grafico = px.scatter(x=X_plano_saude.ravel(),  y=y_plano_saude)
grafico.add_scatter(x=X_plano_saude.ravel(), y = previsoes, name='Regressão Linear')
grafico.show()


# MÉTRICAS

score = regressao_linear.score(X_plano_saude, y_plano_saude)
print(score)

print(regressao_linear.predict([[40]]))


visualizador = ResidualsPlot(regressao_linear)
visualizador.fit(X_plano_saude, y_plano_saude)
visualizador.score(X_plano_saude, y_plano_saude)
visualizador.poof()