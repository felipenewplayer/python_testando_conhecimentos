import bs4 as bs
import urllib.request
import spacy
from spacy.matcher import PhraseMatcher
from IPython.display import display, HTML

# -------- WEB SCRAPING ----------
url = "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial"
headers = {"User-Agent": "Mozilla/5.0"}
req = urllib.request.Request(url, headers=headers)
dados = urllib.request.urlopen(req).read()

dados_html = bs.BeautifulSoup(dados, 'html.parser')

paragrafos = dados_html.find_all('p')
conteudo = ""
for p in paragrafos:
    conteudo += p.text

conteudo = conteudo.lower()

# -------- PROCESSAMENTO COM SPACY ----------
pln = spacy.load("pt_core_news_sm")

string = 'turing'
token_pesquisa = pln(string)

matcher = PhraseMatcher(pln.vocab)
matcher.add("SEARCH", [token_pesquisa])

doc = pln(conteudo)
matches = matcher(doc)

print(matches[0])  # Exibe o primeiro match encontrado
print(doc[130:131], doc[130-5:130+5])  # Exibe o token encontrado e o contexto


numeros_palavras = 50
display(HTML(f"<h3>Contexto da palavra '{string}':</h3>"))