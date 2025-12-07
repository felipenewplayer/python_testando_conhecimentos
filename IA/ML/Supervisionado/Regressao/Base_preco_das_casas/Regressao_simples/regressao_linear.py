import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

base_houses_prices = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Regressao\Base_preco_das_casas\house_prices.csv')

# base_houses_prices = base_houses_prices.drop('date', axis=1)
print(base_houses_prices.columns)
print(base_houses_prices.head())
print(base_houses_prices.describe())
print(base_houses_prices.isnull().sum())

X_casa = base_houses_prices.iloc[:,5:6].values
y_casa = base_houses_prices.iloc[:, 2].values

# figura = plt.figure(figsize=(20,20))
# sns.heatmap(base_houses_prices.corr(), annot=True, cmap='viridis')
# plt.show()  

X_casa_treinamento, X_casa_teste, y_casa_treinamento, y_casa_teste = train_test_split(X_casa, y_casa, test_size=0.3, random_state=0)

regressor_simples = LinearRegression()
regressor_simples.fit(X_casa_treinamento, y_casa_treinamento)

print(regressor_simples.intercept_) #b0
print(regressor_simples.coef_) #b1

print(regressor_simples.score(X_casa_teste, y_casa_teste))
print(regressor_simples.score(X_casa_treinamento, y_casa_treinamento))

previsoes = regressor_simples.predict(X_casa_teste)

print(previsoes)
print(y_casa_teste)

print(mean_absolute_error(y_casa_teste, previsoes))  # MAE
print(mean_squared_error(y_casa_teste, previsoes))  # MSE
np.sqrt(mean_squared_error(y_casa_teste, previsoes))  # RMSE

fig = px.scatter(
    x=X_casa_teste.ravel(),
    y=y_casa_teste,
    title="Regressão Linear - sqft_lot vs bedrooms"
)
fig.add_scatter(
    x=X_casa_teste.ravel(),
    y=previsoes,
    mode="lines",
    name="Linha de Regressão"
)
fig.show()
