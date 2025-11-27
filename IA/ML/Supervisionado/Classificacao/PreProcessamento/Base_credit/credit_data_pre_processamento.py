import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

base_credit = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit_data.csv')

print(base_credit.head())
print(base_credit.describe())

print(base_credit[base_credit['income'] >= 69995.685578 ])


# Contagem de quantos registro exitem em casa uma das classe

print(np.unique(base_credit['default'], return_counts=True))

#Graficos com dados
# sns.countplot(x='default', data=base_credit)
# plt.show()
# plt.hist(base_credit['age'])
# plt.show()
# plt.hist(base_credit['income'])
# plt.show()
# plt.hist(base_credit['loan'])
# plt.show()

print(base_credit['age'][base_credit['age'] > 0 ].mean())
base_credit.loc[base_credit['age'] < 0, 'age' ] = 40.92
print(base_credit.loc[base_credit['age'] < 0 ])



# Tratamento de valores faltantes
print(base_credit.isnull().sum())
print(base_credit.loc[pd.isnull(base_credit['age'])])

base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
print(base_credit.loc[pd.isnull(base_credit['age'])])


#Previsores e calsses

X_credit = base_credit.iloc[:, 1:4].values
print(X_credit)

y_credit= base_credit.iloc[:, 4].values
print(y_credit)


# Escalonamento de atributos


scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())  

# Treinamento e teste

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)

print(X_credit_treinamento.shape)
print(y_credit_treinamento.shape)
print('Teste',X_credit_teste.shape,y_credit_teste.shape)


with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'wb') as f:
    pickle.dump([X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste], f)