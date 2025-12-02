from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier    
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb') as f:
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = pickle.load(f)

print(X_credit_train.shape,y_credit_train.shape)
print(X_credit_test.shape,y_credit_test.shape)

# Na Cross-Validation, iremos testar diversos hiperparâmetros para encontrar a melhor combinação para o nosso modelo, utilizando o GridSearchCV, previsamos 
# Concatenar.

X_credit = np.concatenate((X_credit_train, X_credit_test), axis=0)
print(X_credit.shape)
y_credit = np.concatenate((y_credit_train, y_credit_test), axis=0)
print(y_credit)

parametros = {'criterion':['gini','entropy'],
              'splitter':['best','random'],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,5,10],}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultao = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultao)