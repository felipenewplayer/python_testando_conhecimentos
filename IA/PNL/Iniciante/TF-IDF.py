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

dados = {
    "texto": [
        # ALEGRIA
        "estou muito feliz hoje",
        "que dia maravilhoso",
        "recebi uma ótima notícia",
        "estou sorrindo o dia inteiro",
        "me sinto em paz comigo mesmo",
        "essa conquista me deixou muito feliz",
        "estou animado com o resultado",
        "finalmente deu tudo certo",
        "me sinto grato pela ajuda",
        "esse momento é especial",
        "tive um dia incrível",
        "estou satisfeito com meu trabalho",
        "que alegria estar aqui",
        "isso me deixa muito contente",
        "estou confiante e tranquilo",

        # MEDO
        "estou com muito medo",
        "isso está me assustando",
        "não me sinto seguro aqui",
        "estou com receio do que pode acontecer",
        "essa situação é perigosa",
        "meu coração está acelerado de medo",
        "tenho medo de errar",
        "isso me dá arrepios",
        "estou nervoso e apreensivo",
        "algo ruim pode acontecer",
        "não quero ficar sozinho aqui",
        "essa notícia me deixou assustado",
        "estou inseguro com essa decisão",
        "tenho um mau pressentimento",
        "isso é realmente assustador"
    ],
    "emocao": [
        "alegria"] * 15 + ["medo"] * 15
}


df = pd.DataFrame(dados)
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