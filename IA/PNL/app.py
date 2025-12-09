import bs4 as bs
import urllib.request
import nltk
import spacy


# POS (part-of-speech) atribui para cada palavra de um texto sua classe gramatical
# Exemplo: substantivo, verbo, adjetivo, advérbio, pronome
pnl = spacy.load("pt_core_news_sm")
print(pnl)

documento = pnl("Estou aprendendo processamento de linguagem natural, em São Paulo.")

for token in documento:
    print(token.text, token.pos_)