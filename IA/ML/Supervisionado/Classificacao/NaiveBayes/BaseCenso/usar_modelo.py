import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
with open('dados.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

print(X_treino.shape)
print(y_teste.shape)



