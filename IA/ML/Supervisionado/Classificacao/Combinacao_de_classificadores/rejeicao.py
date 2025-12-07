import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb')as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)

X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

rede_neural = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\rede_neural_finalizado.sav', 'rb'))

svm = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\svm_finalizado.sav', 'rb'))

arvore_decisao = pickle.load(open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos\arvore_decisao_finalizado.sav', 'rb'))

novo_registro = X_credit[0].reshape(1, -1)
resposta_rede_neural= rede_neural.predict(novo_registro)
svm_resposta = svm.predict(novo_registro)
arvore_decisao_resposta = arvore_decisao.predict(novo_registro)


probabilidade_rede_neural = rede_neural.predict_proba(novo_registro)
print(f'Probabilidade Rede Neural: {probabilidade_rede_neural}')

confiaca_rede_neural = probabilidade_rede_neural.max()
print(f'Confiança Rede Neural: {confiaca_rede_neural}')

proprabilidade_arvore_decisao = arvore_decisao.predict_proba(novo_registro)
print(f'Probabilidade Árvore de Decisão: {proprabilidade_arvore_decisao}')
confianca_arvore_decisao = proprabilidade_arvore_decisao.max()
print(f'Confiança Árvore de Decisão: {confianca_arvore_decisao}')

probalidade_svm = svm.predict_proba(novo_registro)
print(f'Probabilidade SVM: {probalidade_svm}')
confianca_svm = probalidade_svm.max()   

paga = 0
nao_paga = 0
confiana_minimia = 0.99999
algoritmos_confiaveis = 0

if confiaca_rede_neural >= confiana_minimia:
 algoritmos_confiaveis += 1
 if  resposta_rede_neural[0] == 1:
    nao_paga += 1
 else:
    paga += 1
if confianca_svm >= confiana_minimia:
 algoritmos_confiaveis += 1
 if svm_resposta[0] == 1:
    nao_paga += 1
 else:
    paga += 1   
if svm_resposta[0] == 1:
    nao_paga += 1
else:
    paga += 1
    
if confianca_arvore_decisao >= confiana_minimia:
 algoritmos_confiaveis += 1
if arvore_decisao_resposta[0] == 1:
    nao_paga += 1
else:
    paga += 1
    
if paga > nao_paga:
    print('O cliente provavelmente pagará o empréstimo, baseado em algoritmos confiáveis.'.format(algoritmos_confiaveis))
elif paga == nao_paga:
    print('Empate entre algoritmos confiáveis.'.format(algoritmos_confiaveis))
else:
    print('O cliente provavelmente não pagará o empréstimo, baseado em algoritmos confiáveis.'.format(algoritmos_confiaveis))
    