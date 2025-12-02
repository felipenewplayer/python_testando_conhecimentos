from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier    
from sklearn.model_selection import GridSearchCV
import pickle

with open(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Classificacao\PreProcessamento\Base_credit\credit.pkl', 'rb') as f:
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = pickle.load(f)

print(X_credit_train.shape,y_credit_train.shape)
print(X_credit_test.shape,y_credit_test.shape)