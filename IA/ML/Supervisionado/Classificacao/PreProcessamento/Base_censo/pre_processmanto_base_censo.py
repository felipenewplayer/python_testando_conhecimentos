import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import pickle
base_census = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_censo\census.csv')

print(base_census.head(5))
print(base_census.describe())

print(base_census.isnull().sum())
print(np.unique(base_census['income'], return_counts=True))
# sns.countplot(x=base_census['income'])
# plt.show()

# Divisão entre previsores e classe

X_census = base_census.iloc[:,0:14].values
print(f'\n X_Census',X_census[0])

y_census = base_census.iloc[:, 14].values
print(f'\n Y_Census',y_census)

#Tratamento de dados categóricos

cat_col = [1,3,5,6,7,8,9,13]

columns_tranformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), cat_col)
], remainder='passthrough')

X_census = columns_tranformer.fit_transform(X_census).toarray()
print(X_census.shape)

# Escalonamento de dados

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
print(X_census[0])

# Treinamento e teste


X_census_train, X_census_test, y_census_train, y_census_test = train_test_split(
    X_census, y_census, test_size=0.15, random_state=0
)

print(X_census_train.shape, y_census_train.shape)
print(X_census_test.shape, y_census_test.shape)


with open('census_pre_processado.pkl', 'wb') as f:
    pickle.dump([X_census_train, X_census_test, y_census_train, y_census_test], f)
    