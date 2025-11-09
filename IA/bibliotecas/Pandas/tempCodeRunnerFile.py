import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\bibliotecas\Pandas\athlete_events.csv')

dados['Name'] = dados['Name'].str.split().str[0]
dados.drop(columns=['Sport','Event'],inplace=True)

plt.figure(figsize=(10,6))
plt.hist(x=dados['Age'].dropna(), bins=20, edgecolor='purple')
plt.title('Relação de Idades')
plt.xlabel('Idade')
plt.show()


grafico = px.scatter_matrix(dados,dimensions=['Age'])
