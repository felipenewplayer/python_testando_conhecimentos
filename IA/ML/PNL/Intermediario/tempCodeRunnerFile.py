import gensim
import spacy
import string
from spacy.lang.pt.stop_words import STOP_WORDS

from gensim.models import Word2Vec

nlp = spacy.load("pt_core_news_sm")
textos = [
    "estou muito feliz hoje",
    "estou sentindo alegria",
    "feliz e alegre com a vida",
    "sinto alegria e felicidade",
    "estou feliz e cheio de alegria",
    "essa alegria me deixou feliz"
    ]

def preprocess(texto):

    texto = str(texto).lower()
    doc = nlp(texto)
    
    tokens = [

        token.text for token in doc
        if token.text not in string.punctuation
        and token.text not in STOP_WORDS
        and token.is_alpha
    ]
    
    return tokens


sentencas = [preprocess(t) for t in textos]
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

print(modelo.wv.index_to_key)

similariedade_feliz_alegria = modelo.wv.similarity("feliz","alegria")
print(similariedade_feliz_alegria)