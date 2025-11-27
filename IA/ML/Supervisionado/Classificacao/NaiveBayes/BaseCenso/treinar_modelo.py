import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle

# Leitura dos dados
dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\NaiveBayes\BaseCenso\credit_data.csv')
print(dados.head(6))

# Tratar NaNs
dados['age'] = dados['age'].fillna(dados['age'].mean())
dados['income'] = dados['income'].fillna(dados['income'].mean())
dados['loan'] = dados['loan'].fillna(dados['loan'].mean())

# Separar X e y
X_dados = dados.iloc[:, 1:4].values
y_dados = dados.iloc[:, 4].values

# Escalonamento
scaler = StandardScaler()
X_dados = scaler.fit_transform(X_dados)

# Dividir treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_dados, y_dados, test_size=0.25, random_state=0)

# Salvar dados processados
with open('dados.pkl', 'wb') as f:
    pickle.dump((X_treino, X_teste, y_treino, y_teste), f)

# Treinar modelo
modelo = GaussianNB()
modelo.fit(X_treino, y_treino)

# Previsão
previsao = modelo.predict(X_teste)
print(previsao)


with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\NaiveBayes\BaseDados\dados.pkl', 'wb') as f:
    pickle.dump((X_treino,y_treino), f)
    

