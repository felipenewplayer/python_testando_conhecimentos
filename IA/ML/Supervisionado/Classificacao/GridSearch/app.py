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

# print(X_credit_train.shape,y_credit_train.shape)
# print(X_credit_test.shape,y_credit_test.shape)

# # Na Cross-Validation, iremos testar diversos hiperparâmetros para encontrar a melhor combinação para o nosso modelo, utilizando o GridSearchCV, previsamos 
# # Concatenar.

X_credit = np.concatenate((X_credit_train, X_credit_test), axis=0)
# print(X_credit.shape)
y_credit = np.concatenate((y_credit_train, y_credit_test), axis=0)
# print(y_credit)


####### Ávore de Decisão #######

# parametros = {'criterion':['gini','entropy'],
#               'splitter':['best','random'],
#             'min_samples_split':[2,5,10],
#             'min_samples_leaf':[1,5,10],}

# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=parametros)
# grid_search.fit(X_credit, y_credit)
# melhores_parametros = grid_search.best_params_
# melhor_resultao = grid_search.best_score_
# print(melhores_parametros)
# print(melhor_resultao)

######## Random Forest ########

# parametros = {'criterion':['gini','entropy'],
#              'n_estimators':[10,40,100,150],
#              'min_samples_split':[2,5,10],
#              'min_samples_leaf':[1,5,10],}

# grid_search = GridSearchCV(estimator=RandomForestClassifier(),param_grid=parametros)
# grid_search.fit(X_credit, y_credit)
# melhores_parametros = grid_search.best_params_
# melhor_resultado = grid_search.best_score_
# print(melhores_parametros)
# print(melhor_resultado)

######## KNN ########

# parametros = { 'n_neighbors':[3,5,10,20],
#               'p':[1,2],}

# grid_search = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=parametros)
# grid_search.fit(X_credit, y_credit)
# melhores_parametros = grid_search.best_params_
# melhor_resultado = grid_search.best_score_
# print(melhores_parametros)
# print(melhor_resultado)

######## Logistic Regression ########

# parametros = { 'tol':[0.0001,0.00001,0.000001],
#               'C':[1.0,1.5, 2.0],
#               'solver':['lbfgs','saga','sag'],}

# grid_search = GridSearchCV(estimator=LogisticRegression(),param_grid=parametros)
# grid_search.fit(X_credit, y_credit)
# melhores_parametros = grid_search.best_params_
# melhor_resultado = grid_search.best_score_
# print(melhores_parametros)
# print(melhor_resultado)


######## SVM ########

# parametros = { 'tol':[0.0001,0.00001,0.000001],
#               'C':[1.0,1.5, 2.0],
#               'kernel':['rbf','linear','poly', 'sigmoid'],}

# grid_search = GridSearchCV(estimator=SVC(),param_grid=parametros)
# grid_search.fit(X_credit, y_credit)
# melhores_parametros = grid_search.best_params_
# melhor_resultado = grid_search.best_score_
# print(melhores_parametros)
# print(melhor_resultado)


######## Redes Neurais ########

# parametros = {'activation':['logistic','tanh','relu','logistic'],
#               'solver':['adam','sgd'],
#               'batch_size':[10,56],}

# grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros)
# grid_search.fit(X_credit, y_credit)
# melhores_parametros = grid_search.best_params_
# melhor_resultado = grid_search.best_score_
# print(melhores_parametros)
# print(melhor_resultado)


######## CrossValidation ########

from sklearn.model_selection import cross_val_score, KFold

resultados_arvores = []
resultados_random_forest = []
resultados_knn = []
resultados_logist = []
resultados_svc = []
resultados_rede_neural = []

for i in range(30):
    print(f"Rodada {i}")

    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    # Árvore
    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, 
                                    min_samples_split=5, splitter='best')
    score = cross_val_score(arvore, X_credit, y_credit, cv=kfold).mean()
    resultados_arvores.append(score)
    print(f"Árvore = {score}")

    # Random Forest
    random_florest = RandomForestClassifier(criterion='entropy', 
                                            min_samples_leaf=1, 
                                            min_samples_split=5, 
                                            n_estimators=10)
    score = cross_val_score(random_florest, X_credit, y_credit, cv=kfold).mean()
    resultados_random_forest.append(score)
    print(f"Random Forest = {score}")

    # KNN
    knn = KNeighborsClassifier()
    score = cross_val_score(knn, X_credit, y_credit, cv=kfold).mean()
    resultados_knn.append(score)
    print(f"KNN = {score}")

    # Regressão Logística
    logistica = LogisticRegression(C=1.0, solver='lbfgs', tol=0.001, max_iter=500)
    score = cross_val_score(logistica, X_credit, y_credit, cv=kfold).mean()
    resultados_logist.append(score)
    print(f"Logística = {score}")

    # SVM
    svm = SVC(kernel='rbf', C=2.0)
    score = cross_val_score(svm, X_credit, y_credit, cv=kfold).mean()
    resultados_svc.append(score)
    print(f"SVM = {score}")

    # Rede Neural
    rede_neural = MLPClassifier(activation='relu', batch_size=56, 
                                solver='adam', max_iter=500)
    score = cross_val_score(rede_neural, X_credit, y_credit, cv=kfold).mean()
    resultados_rede_neural.append(score)
    print(f"Rede Neural = {score}")
 