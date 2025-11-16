import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Base de dados
df = pd.DataFrame({
    'Nome': ['Felipe','Rafa','Jona'],
    'Divida':[10000,5000,1000],
    'Idade': [20,32,41],
    'Altura': [100,120,140],
    'Pago':[0,1,0]
})


x_variable = df.iloc[:, 1: 4].values
print(x_variable)
print(type(x_variable))

y_variable = df.iloc[:, 4].values
print(y_variable)

print(x_variable[:, 0].min())
print(x_variable[:, 0].max())
print(x_variable[:,1].min())

scaler = StandardScaler()
x_variable = scaler.fit_transform(x_variable)
print(x_variable[:, 0].min(),x_variable[:, 1].min(),x_variable[:,2].min())
print(x_variable[:, 0].max(),x_variable[:, 1].max(),x_variable[:,2].max())
