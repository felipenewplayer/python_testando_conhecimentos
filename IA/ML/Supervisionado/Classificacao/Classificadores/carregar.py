import pickle
import numpy as np

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb')as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)
    

X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

rede_neural = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\rede_neural_finalizado.sav', 'rb'))
svm = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\svm_finalizado.sav', 'rb'))
arvore_decisao = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\arvore_decisao_finalizado.sav', 'rb'))

novo_registro = X_credit[0].reshape(1, -1)
print(novo_registro.shape)
previsao_rede_neural = rede_neural.predict(novo_registro)
previsao_svm = svm.predict(novo_registro)
previsao_arvore_decisao = arvore_decisao.predict(novo_registro)
print(f'Rede Neural: {previsao_rede_neural[0]}')
print(f'SVM: {previsao_svm[0]}')
print(f'Árvore de Decisão: {previsao_arvore_decisao[0]}')
