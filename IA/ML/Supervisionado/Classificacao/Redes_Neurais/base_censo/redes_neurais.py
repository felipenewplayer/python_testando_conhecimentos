from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_censo\base_censo.pkl', 'rb') as f:
    X_censo_train, X_censo_test, y_censo_train, y_censo_test = pickle.load(f)
    
rede_neural_censo = MLPClassifier(max_iter=1000, verbose=True , tol= 0.000010, solver='adam',activation='relu',hidden_layer_sizes=(55,55))
rede_neural_censo.fit(X_censo_train, y_censo_train)
previsoes = rede_neural_censo.predict(X_censo_test)
print(previsoes)
print(y_censo_test)
acuracia = accuracy_score(y_censo_test, previsoes)
print(f'ACURÁCIA: {acuracia}')
cm = ConfusionMatrix(rede_neural_censo)
cm.fit(X_censo_train, y_censo_train)
cm.score(X_censo_test, y_censo_test)
cm.show()
relatorio = classification_report(y_censo_test, previsoes)
print(relatorio)  