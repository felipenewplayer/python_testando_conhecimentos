import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Base de dados
df = pd.DataFrame({
    'Nome': ['Felipe','Rafa','Jona','Thay','Marcos','José','Souza','Luis'],
    'Idade': [20,32,41,55,66,87,10,50],
    'Altura': [100,120,140,80,180,175,190,173],
    'Peso': [55,60,89,80,75,73,90,100]
})

# OneHotEncoder precisa de 2D -> [['Nome']]
encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False pra mostrar como array
nomes_codificados = encoder.fit_transform(df[['Nome']])

print("\nCodificação OneHotEncoder (nomes):")
print(nomes_codificados)
print("\nCategorias detectadas:", encoder.categories_)

# Agora adicionando os nomes codificados ao dataframe (opcional)
df_codificado = pd.concat([df, pd.DataFrame(nomes_codificados, columns=encoder.get_feature_names_out(['Nome']))], axis=1)
print("\nDataFrame com nomes codificados:")
print(df_codificado)

# Variáveis para regressão (Idade -> Peso)
x = np.array(df['Idade']).reshape(-1,1)
y = np.array(df['Peso']).reshape(-1,1)

# Padronização
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaler = scaler_x.fit_transform(x)
y_scaler = scaler_y.fit_transform(y)

# Modelo
modelo = LinearRegression()
modelo.fit(x_scaler, y_scaler)

# Entrada do usuário
dado_inputado = int(input("\nDigite uma idade: "))
idade_nova = np.array([[dado_inputado]])
idade_nova_scaler = scaler_x.transform(idade_nova)

# Previsão
previsao_scaler = modelo.predict(idade_nova_scaler)
previsao_real = scaler_y.inverse_transform(previsao_scaler)

print(f"\nPrevisão padronizada: {previsao_scaler}")
print(f"Previsão real (peso estimado): {previsao_real[0][0]:.2f} kg")
