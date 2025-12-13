import spacy
import pandas as pd
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import string


nlp = spacy.load("pt_core_news_sm")
base_de_dados = pd.read_csv(r"C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\PNL\dados.csv")
df = pd.DataFrame(base_de_dados)
print(df)

def preprocess(texto):
    texto = str(texto).lower()
    doc = nlp(texto)
    
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in string.punctuation
        and token.text not in STOP_WORDS
        and token.text.isalpha()
    ]
    
    return" ".join(tokens)

df["texto_limpo"] = df["texto"].apply(preprocess)

X_texto = df["texto_limpo"]
y = df["emocao"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X_texto)
print("\nMatriz TF-IDF:")
print(X.toarray())

print("\nVocabulário:")
print(vectorizer.get_feature_names_out())


X_train,X_test,y_train,y_test = train_test_split(X
                                                 ,y
                                                 ,
                                                 test_size=0.25,
                                                 random_state=42)

modelo = LogisticRegression(max_iter=1000,    class_weight="balanced"
)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print(f"Accuracy:{acc} ")



print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



novos_textos = [
    "meu coração disparou de susto",
    "estou tranquilo e satisfeito",
    "essa situação me deixa inseguro",
    "me sinto realizado com isso"
]


novos_limpos = [preprocess(t) for t in novos_textos]
X_novo = vectorizer.transform(novos_limpos)

print(modelo.predict(X_novo))   

scores = cross_val_score(
    modelo, X, y, cv=5, scoring="f1_macro"
)

print(scores)
print("F1_médio: ",  scores.mean())