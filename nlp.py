import spacy
import re, string

nlp = spacy.load("pt_core_news_sm")
text = "Estou aprendendo processamento de linguagem natural com spacy!!!"
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","", text)
    text = re.sub(r"\s+"," ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t) > 2]
    return" ".join(tokens)

print(preprocess_text(text))