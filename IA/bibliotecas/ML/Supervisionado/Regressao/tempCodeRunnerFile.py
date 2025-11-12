
# x = np.array(df['Idade']).reshape(-1,1)
# y = np.array(df['Peso']).reshape(-1,1)

# scaler_x = StandardScaler()
# scaler_y = StandardScaler()
# x_scaler = scaler_x.fit_transform(x)
# y_scaler = scaler_y.fit_transform(y)

# print(y_scaler)
# print(x_scaler)

# modelo = LinearRegression()
# modelo.fit(x_scaler,y_scaler)

# dado_inputado = int(input("Digite uma idade : "))
# idade_nova = np.array([[dado_inputado]])
# idade_nova_scaler = scaler_x.transform(idade_nova)

# previsao_scaler = modelo.predict(idade_nova_scaler)
# previsao_real = scaler_y.inverse_transform(previsao_scaler)


# print(f"Previsão padronizada: {previsao_scaler}")
# print(f"Previsão real (peso estimado): {previsao_real[0][0]:.2f} kg")