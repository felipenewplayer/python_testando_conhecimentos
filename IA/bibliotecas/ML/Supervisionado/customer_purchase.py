import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\bibliotecas\ML\Supervisionado\Customer Purchase.csv')

dados.drop(columns='Customer ID', inplace=True)

print(dados.head(5))
print(dados.info())
print(dados.isnull().sum())
print(dados.describe())



print(np.unique(dados['Age']))
print(np.unique(dados['Education']))
print(np.unique(dados['Gender']))
print(np.unique(dados['Review']))

# sns.countplot(x=dados['Age'])
# plt.hist(x=dados['Age'])
# plt.show()

# grafico = px.treemap(dados, path=['Education','Age','Review'])
# grafico.show()

X_dados = dados.iloc[:, 0:4].values

print(X_dados)
print(X_dados[0])

Y_dados = dados.iloc[:, 4].values
print(Y_dados)

encoder = LabelEncoder()

label_encoder_gender = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_review = LabelEncoder()

X_dados[:,1] = label_encoder_gender.fit_transform(X_dados[:,1])
X_dados[:,2] = label_encoder_education.fit_transform(X_dados[:,2])
X_dados[:,3] = label_encoder_review.fit_transform(X_dados[:,3])

print(X_dados[0])


onehotencoder_dados = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(sparse_output=False),[1,2,3])], remainder='passthrough')

X_dados = onehotencoder_dados.fit_transform(X_dados)
print(X_dados)
print(X_dados.shape)


scaler = StandardScaler()
X_dados = scaler.fit_transform(X_dados)
print(X_dados[0])



