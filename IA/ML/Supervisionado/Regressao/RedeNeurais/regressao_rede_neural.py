import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
import plotly.express as px

base_de_dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\plano_saude2.csv')

X_plano_saude = base_de_dados.iloc[:,0].values.reshape(-1,1)
y_planos_saude = base_de_dados.iloc[:,1].values.reshape(-1,1)

# ---- SCALERS SEPARADOS ----
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_plano_saude_scaled = scaler_x.fit_transform(X_plano_saude)
y_planos_saude_scaled = scaler_y.fit_transform(y_planos_saude).ravel()

# ---- RNA ----
regressor_rna = MLPRegressor(max_iter=1000, random_state=1)
regressor_rna.fit(X_plano_saude_scaled, y_planos_saude_scaled)

score_rna = regressor_rna.score(X_plano_saude_scaled, y_planos_saude_scaled)
print(f'Score da RNA: {score_rna}')

# ---- Gráfico ----
grafico = px.scatter(
    x=X_plano_saude.ravel(),
    y=y_planos_saude.ravel(),
    color_discrete_sequence=['blue'],
    title='Plano de Saúde x Idade'
)

y_pred_scaled = regressor_rna.predict(X_plano_saude_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

grafico.add_scatter(
    x=X_plano_saude.ravel(),
    y=y_pred,
    mode='lines',
    line=dict(color='red'),
    name='Regressão com RNA'
)

grafico.show()

# ---- Previsão nova ----
novo = np.array([[40]])
novo_scaled = scaler_x.transform(novo)
print(novo_scaled)

pred_scaled = regressor_rna.predict(novo_scaled)
pred_final = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))
print(pred_final)
