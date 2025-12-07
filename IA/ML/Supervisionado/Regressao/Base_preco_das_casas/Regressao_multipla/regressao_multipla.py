import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_casas = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Regressao\Base_preco_das_casas\house_prices.csv')

print(base_casas.head())

X_casas = base_casas.iloc[:, 3:19].values
y_casas = base_casas.iloc[:, 2].values
print(X_casas)
print(y_casas)


X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste=train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

regressaor_multiplo = LinearRegression()
regressaor_multiplo.fit(X_casas_treinamento, y_casas_treinamento)

print(regressaor_multiplo.intercept_) # b0
print(regressaor_multiplo.coef_)      # b1, b2, b3, ..., bn
print(regressaor_multiplo.score(X_casas_treinamento, y_casas_treinamento)) # R²
print(regressaor_multiplo.score(X_casas_teste, y_casas_teste))         # R²

print(y_casas_teste)
previsoes = regressaor_multiplo.predict(X_casas_teste)
print(previsoes)

print(mean_absolute_error(y_casas_teste, previsoes)) # MAE
print(mean_squared_error(y_casas_teste, previsoes)) # MSE
print(np.sqrt(mean_squared_error(y_casas_teste, previsoes))) # RMSE