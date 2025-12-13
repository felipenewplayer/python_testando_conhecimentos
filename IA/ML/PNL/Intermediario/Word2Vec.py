import spacy
import string
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

base_de_dados = pd.read_csv(r"C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\PNL\dados.csv")

nlp = spacy.load("pt_core_news_sm")

def preprocess(texto):

    texto = str(texto).lower()
    doc = nlp(texto)
    
    tokens = [

        token.text for token in doc
        if token.text not in string.punctuation
        and token.is_alpha
    ]
    
    return tokens


sentencas = [preprocess(t) for t in base_de_dados["texto"]]
print(sentencas)
modelo = Word2Vec(
    sentences=sentencas,
    vector_size=100,   # tamanho do vetor (embedding)
    window=3,          # tamanho do contexto
    min_count=1,       # ignora palavras raras (<1)
    sg=1,              # 1 = Skip-gram | 0 = CBOW
    workers=4,         # núcleos da CPU
    epochs=100
)

vetor_feliz = modelo.wv["feliz"]
print(vetor_feliz)
print(vetor_feliz.shape)

def vetor_frase(tokens):
    vetores = [
        modelo.wv[p]
        for p in tokens
        if p in modelo.wv
    ]
    
    if len(vetores)==0:
        return np.zeros(modelo.vector_size)
    
    return np.mean(vetores, axis=0)
    
X = np.array([vetor_frase(s) for s in sentencas])
y = base_de_dados["emocao"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
modelo_classificador = LogisticRegression(max_iter=1000, class_weight="balanced")
modelo_classificador.fit(X_train, y_train)

y_pred = modelo_classificador.predict(X_test)
acuracidade = accuracy_score(y_test,y_pred)

print(f"acuracidade:  {acuracidade}")
print(f"classificação:  {classification_report(y_test,y_pred)}")