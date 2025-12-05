import pickle
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb')as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)
    
X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)


rede_neural = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\rede_neural_finalizado.sav', 'rb'))

svm = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\svm_finalizado.sav', 'rb'))

arvore_decisao = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\arvore_decisao_finalizado.sav', 'rb'))

novo_registro = X_credit[1999].reshape(1, -1)

resposta_rede_neural= rede_neural.predict(novo_registro)
svm_resposta = svm.predict(novo_registro)
arvore_decisao_resposta = arvore_decisao.predict(novo_registro)

print(f'Rede Neural: {resposta_rede_neural[0]}')
print(f'SVM: {svm_resposta[0]}')
print(f'Árvore de Decisão: {arvore_decisao_resposta[0]}')

paga = 0 
nao_paga = 0

if  resposta_rede_neural[0] == 1:
    nao_paga += 1
else:
    paga += 1
    
if svm_resposta[0] == 1:
    nao_paga += 1
else:
    paga += 1
    
if arvore_decisao_resposta[0] == 1:
    nao_paga += 1
else:
    paga += 1
    
if paga > nao_paga:
    print('O cliente provavelmente pagará o empréstimo.')
elif paga == nao_paga:
    print('Empate.')
else:
    print('O cliente provavelmente não pagará o empréstimo.')
    

    




