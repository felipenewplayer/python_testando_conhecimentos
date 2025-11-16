import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle
dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\NaiveBayes\risco_credito.csv',)

print(dados.head(5))


X_risco_credit = dados.iloc[:,0:4].values # Usa-se .values para converte no formato do numpy array

y_risco_credit = dados.iloc[:,4].values # Usa-se .values para converte no formato do numpy array

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credit[:,0] = label_encoder_historia.fit_transform(X_risco_credit[:,0])
X_risco_credit[:,1] = label_encoder_divida.fit_transform(X_risco_credit[:,1])
X_risco_credit[:,2] = label_encoder_garantia.fit_transform(X_risco_credit[:,2])
X_risco_credit[:,3] = label_encoder_renda.fit_transform(X_risco_credit[:,3]) 

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado/risco_credito.pkl','wb') as f:
    pickle.dump([X_risco_credit,y_risco_credit],f)
    
    
naive_risco_credit = GaussianNB()

    
