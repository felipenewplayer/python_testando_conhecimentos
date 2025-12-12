import pandas as pd
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.util import minibatch
from spacy.training import Example
import string
import random

base_de_dados = pd.read_csv(r"C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\PNL\Classificacao_de_sentimentos_com_spacy\base_treinamento.txt")

print(base_de_dados.head())

# NLP LIMPO PARA TREINAR
nlp = spacy.blank("pt")
textcat = nlp.add_pipe("textcat")
textcat.add_label("ALEGRIA")
textcat.add_label("MEDO")


# NLP SEPARADO PARA PROCESSAR
nlp_pre = spacy.blank("pt")

pontuacoes = string.punctuation

def preprocess(texto):
    texto = str(texto).lower()
    doc = nlp_pre(texto)
    tokens = [
        t.text for t in doc
        if t.text not in pontuacoes
        and t.text not in STOP_WORDS
    ]
    return " ".join(tokens)


base_de_dados['texto'] = base_de_dados["texto"].apply(preprocess)


dados_treinamento = []

for texto, emocao in zip(base_de_dados["texto"], base_de_dados["emocao"]):
    if emocao == "alegria":
        cats = {"ALEGRIA": True, "MEDO": False}
    else:
        cats = {"ALEGRIA":False, "MEDO":True}
    
    dados_treinamento.append((texto, {"cats":cats}))

nlp.initialize()


for epoca in range (20):
    random.shuffle(dados_treinamento)
    losses = {}
    
    for lote in minibatch(dados_treinamento, size = 20):
        exemplos = []
        
        for texto, anotacao in lote:
            doc = nlp.make_doc(texto)
            exemplo = Example.from_dict(doc, anotacao)
            exemplos.append(exemplo)
            
        nlp.update(exemplos, losses=losses)
    print(f"Época {epoca+1}, Losses:{losses}")


nlp.to_disk("modeldo_de_sentimento_com_spacy")
print("Modelo Salvo")


modelo = spacy.load("modeldo_de_sentimento_com_spacy")


textos_teste = [
    "estou muito feliz hoje",
    "isso está me assustando",
    "que dia maravilhoso",
    "estou com muito medo desse lugar"
]

for t in textos_teste:
    doc = modelo(t)
    print(f"\nTexto: {t}")
    print(f"Emoção: {doc.cats}")
    print("Predição:", max(doc.cats, key=doc.cats.get))