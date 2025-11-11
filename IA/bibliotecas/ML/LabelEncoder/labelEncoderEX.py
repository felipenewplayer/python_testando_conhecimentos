import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'Nome':['Felipe', 'Marcos','Rafa','Julia','Larissa'],
    'Idade':[28,20,30,55,34],
    'Divida':[2200, 4500,1000,30,6000]
})

label_encoder_text = LabelEncoder()
df['Nome'] = label_encoder_text.fit_transform(df['Nome'])

x_variable = df.loc[:,['Nome','Divida']].values
y_variable = df.loc[:, ['Idade']].values

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x_variable)
print(x_scaler)

print(x_variable)


# 4️⃣ Treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(x_scaler, y_variable)

# 5️⃣ Prever
y_pred = modelo.predict(x_scaler)
print(y_pred)

# 6️⃣ Visualizar resultado
plt.scatter(df['Divida'], df['Idade'], color='blue', label='Original')
plt.plot(df['Divida'], y_pred, color='red', label='Regressão')
plt.xlabel('Dívida')
plt.ylabel('Idade')
plt.title('Relação entre Dívida e Idade')
plt.legend()
plt.show()


