import pickle
import numpy as np

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb') as f:
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = pickle.load(f)

X_credit = np.concatenate((X_credit_train, X_credit_test), axis=0)
y_credit = np.concatenate((y_credit_train, y_credit_test), axis=0)

print(X_credit.shape)
print(y_credit.shape)


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classificador_rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')
classificador_rede_neural.fit(X_credit_train, y_credit_train)
previsao = classificador_rede_neural.predict(X_credit_test)

classificador_arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
classificador_arvore.fit(X_credit_train, y_credit_train)

classificador_svm = SVC(C=2.0, kernel='rbf')
classificador_svm.fit(X_credit_train, y_credit_train)

pickle.dump(classificador_rede_neural, open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos/rede_neural_finalizado.sav', 'wb'))
pickle.dump(classificador_arvore, open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos/arvore_decisao_finalizado.sav', 'wb'))
pickle.dump(classificador_svm, open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\Classificadores\Classificadores_Salvos/svm_finalizado.sav', 'wb'))

print("Modelos salvos com sucesso!")