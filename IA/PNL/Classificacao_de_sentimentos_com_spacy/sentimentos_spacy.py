## ETAPA 1 importar as bibliotecas

import pandas as pd
import spacy
import string
import random
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS
# ETAPA 2 carregar os dados

base_de_dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\PNL\Classificacao_de_sentimentos_com_spacy\base_treinamento.txt', encoding='utf-8')

# print(base_de_dados.shape)
# print(base_de_dados.head())
# print(base_de_dados.tail())

# sns.countplot(base_de_dados['emocao'], label = 'Contagem')

### ETAPA 3 função para preprocessamento dos dados

pontuacoes = string.punctuation

stop_words = STOP_WORDS
# print(stop_words)

pln = spacy.load('pt_core_news_sm')
# print(pln)  


def preprocessamento(texto):
    texto = texto.lower()
    documento = pln(texto)
    lista = []    
    for token in documento:
        # lista.append(token.text)
        lista.append(token.lemma_)
        
    lista =  [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit   ()])
    return lista

    
teste = "Estou aprendendo pnl 10 90  com spacy!!!"
print(preprocessamento(teste))

