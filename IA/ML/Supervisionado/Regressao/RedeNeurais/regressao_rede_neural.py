import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
import plotly.express as px


base_de_dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\plano_saude2.csv')

X_plano_saude = base_de_dados.iloc[:,0].values.reshape(-1,1)
y_planos_saude = base_de_dados.iloc[:,1].values.reshape(-1,1)
print(X_plano_saude)

scaler = StandardScaler()
X_plano_saude_scaled = scaler.fit_transform(X_plano_saude)
print(X_plano_saude_scaled)
y_planos_saude_scaled = scaler.fit_transform(y_planos_saude).ravel()
print(y_planos_saude_scaled)

regressor_rna = MLPRegressor(max_iter=1000)
regressor_rna.fit(X_plano_saude_scaled,y_planos_saude_scaled)

score_rna = regressor_rna.score(X_plano_saude_scaled,y_planos_saude_scaled)
print(f'Score da RNA: {score_rna}')

grafico = px.scatter(x=X_plano_saude.ravel(), y=y_planos_saude.ravel(), color_discrete_sequence=['blue'], title='Plano de Saúde x Idade')
y_pred_scaled = regressor_rna.predict(X_plano_saude_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
grafico.add_scatter(x=X_plano_saude.ravel(), y=y_pred, mode='lines', line=dict(color='red'), name='Regressão com RNA')
grafico.show()


novo = [[40]]
novo = scaler.transform(novo)
print(novo)
print(regressor_rna.predict(novo)
)