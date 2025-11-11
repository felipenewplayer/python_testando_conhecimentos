import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# Lendo os dados
dados = pd.read_csv(
    r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\bibliotecas\Pandas\athlete_events.csv'
)

# Criando cópia e removendo colunas desnecessárias
filtro = dados.drop(columns=['Sport', 'Event']).copy()

# Preenchendo valores faltantes com a média das colunas
filtro[['Age', 'Height', 'Weight']] = filtro[['Age', 'Height', 'Weight']].fillna(
    filtro[['Age', 'Height', 'Weight']].mean()
)

print(np.unique(filtro['Year']))
# sns.countplot(x= filtro['Year'])
# plt.show()
# plt.hist(x = filtro['Year'])
# plt.show()
grafico = px.treemap(filtro, path=['City','Year'])
grafico.show()


# Arredondando as colunas numéricas
filtro[['Height', 'Weight']] = filtro[['Height', 'Weight']].round(2)
filtro['Age'] = filtro['Age'].round(0).astype(int)

# Exibindo as 5 primeiras linhas
print(filtro.head(5))


X_variable = filtro.loc[:, ['Height','Weight','Year']].values
Y_variable = filtro.loc[:, 'Age'].values

print(f'\nVariável Y : \n{Y_variable}')
print(f'\nVariável X : \n{X_variable}')
print(f'\nTipo de dados : {type(X_variable)}')


scaler = StandardScaler()
x_sclaler =scaler.fit_transform(X_variable)

modelo = LinearRegression()
modelo.fit(x_sclaler, Y_variable)

print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)

nova_entrada = scaler.transform([[180,75,2001]])
predicao = modelo.predict(nova_entrada).astype(int)
print(f'Previsão de idade: {predicao[0]}')
r2 = modelo.score(x_sclaler, Y_variable)
print("R² do modelo:", r2)

scaler_altura= x_sclaler[:,0].min()
scaler_peso = x_sclaler[:,1].min()
scaler_ano = x_sclaler[:,2].min()
print(scaler_altura,scaler_peso,scaler_ano)

# medalhas_por_pais = dados.loc[dados['Medal'].notna()].groupby('Team')['Medal'].count().sort_values(ascending=False)
# print(medalhas_por_pais.head(5))

# medalhas_ano = dados.loc[dados['Medal'].notna()].groupby('Year')['Medal'].count().sort_values(ascending=False)
# print(medalhas_ano)

# tabela = dados.loc[dados['Medal'].notna()].pivot_table(
#     index='Team',
#     columns='Medal',
#     values='Name',
#     aggfunc='count',
#     fill_value=0
# )

# print(tabela.head(10))


# grafico = px.bar(
#     medalhas_por_pais.head(10).reset_index(),
#     x='Team',
#     y='Medal',
#     title='Top 10 Países com Mais Medalhas',
#     color='Team'
# )
# grafico.show()





