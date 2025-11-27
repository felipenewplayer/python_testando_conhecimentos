from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb') as f:
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = pickle.load(f)


rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True , tol= 0.0000100, solver='adam',activation='relu',hidden_layer_sizes=(2,2))
rede_neural_credit.fit(X_credit_train, y_credit_train)

previsoes = rede_neural_credit.predict(X_credit_test)
print(previsoes)
print(y_credit_test)
acuracia = accuracy_score(y_credit_test, previsoes)
print(f'ACURÁCIA: {acuracia}')
cm = ConfusionMatrix(rede_neural_credit)
cm.fit(X_credit_train, y_credit_train)
cm.score(X_credit_test, y_credit_test)
cm.show()

relatorio = classification_report(y_credit_test, previsoes)
print(relatorio)