import pandas as pd 
import spacy
import string
import numpy as np
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

base_de_dados = pd.read_csv(r"C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\PNL\dados.csv")

nlp = spacy.load("pt_core_news_sm")

def preprocess(texto):
    texto = str(texto).lower()
    doc = nlp(texto)
    
    tokens = [
        token.text for token in doc
        if token.text not in string.punctuation
        and token.text.isalpha
    ]

    return tokens

sentencas= [preprocess(t) for t in base_de_dados["texto"]]
print(sentencas)

modelo_w2v = Word2Vec(
    sentences=sentencas,
    window=3,
    vector_size=100,
    min_count=4,
    epochs=50
)

def vetor_frase(tokens):
    vetores = [
        modelo_w2v.wv[p]
        for p in tokens
        if p in modelo_w2v.wv
    ]
    
    
    if len(vetores)==0:
        return np.zeros(modelo_w2v.vector_size)
    
    return np.mean(vetores, axis=0)


X = np.array([vetor_frase(s) for s in sentencas ])
y = base_de_dados["emocao"].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)


mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = mlp.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
