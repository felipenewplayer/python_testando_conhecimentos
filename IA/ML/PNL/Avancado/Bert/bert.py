import pandas as pd 
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score


#1 ----- Carregar o dados que serão testados e treinados
base_de_dados = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\PNL\dados.csv')

#2 ------   Converter labels para números
#BERT não aceita string como rótulo

label_map = {"alegria": 0, "medo": 1}
base_de_dados["label"] = base_de_dados["emocao"].map(label_map)

#3 ------  Train / Test split

X_train, X_test, y_train, y_test = train_test_split(
    base_de_dados["texto"].to_list(),
    base_de_dados["label"].to_list(),
    test_size=0.25,
    random_state=42,
)

#4 ---- Tokenizer BERT

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=64)

#5 ---- Dataset PyTorch

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, y_train)
test_dataset = EmotionDataset(test_encodings, y_test)


# 6 ---- Modelo BERT

modelo = BertForSequenceClassification.from_pretrained(
        "neuralmind/bert-base-portuguese-cased",
        num_labels=2
)


# 6 ---- Métricas

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }
    

# Treinamento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    save_strategy="no",
    report_to="none"
)


trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()



# 7 ----- Avaliação
trainer.evaluate()