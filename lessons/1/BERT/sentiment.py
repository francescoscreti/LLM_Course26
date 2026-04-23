# bert_demo.py
# Obiettivo: usare BERT pre-addestrato per classificare le stesse frasi
# del corpus italiano — e confrontare con il modello che abbiamo costruito.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------
# 1. Modello e tokenizer
# ------------------------------------------------------------------

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
# modello BERT multilingue fine-tuned su recensioni in 6 lingue
# incluso l'italiano — output: 5 classi (1 stella -> 5 stelle)
# è uno dei modelli più scaricati su HuggingFace per il sentiment

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# AutoTokenizer carica automaticamente il tokenizer giusto per questo modello
# gestisce vocabolario, regole di tokenizzazione e token speciali
# aggiunge [CLS] e [SEP] automaticamente — noi lo facevamo a mano in data.py

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# scarica i pesi pre-addestrati dal Hub (~700MB la prima volta)
# le volte successive usa la cache locale — nessun download

model.eval()
# disattiva dropout — siamo in inferenza, non in training
# equivalente al model.eval() che abbiamo usato in train.py


# ------------------------------------------------------------------
# 2. Mappatura stelle -> binario
# ------------------------------------------------------------------

def stelle_a_binario(indice_classe: int) -> int:
    """
    nlptown restituisce 5 classi (0=1stella, 1=2stelle, ..., 4=5stelle)
    le mappiamo a binario per confrontare con il nostro corpus:
      0-1 (1-2 stelle) -> negativo (0)
      2-4 (3-5 stelle) -> positivo (1)

    Perché 3 stelle -> positivo?
    È una scelta — potresti anche escluderle o trattarle come neutro.
    Per semplicità le consideriamo positive.
    """
    return 0 if indice_classe <= 2 else 1

def indice_a_stelle(indice_classe: int) -> str:
    # converte l'indice (0-4) nella stringa leggibile ("1 stella", ecc.)
    stelle = indice_classe + 1
    return f"{stelle} {'stella' if stelle == 1 else 'stelle'}"


# ------------------------------------------------------------------
# 3. Le stesse frasi del nostro corpus
# ------------------------------------------------------------------

frasi = [
    ("il film è bellissimo davvero",            1),
    ("storia noiosa e recitazione pessima",      0),
    ("mi ha emozionato moltissimo",              1),
    ("non lo consiglio a nessuno",               0),
    ("capolavoro assoluto del cinema italiano",  1),
    ("una perdita di tempo totale",              0),
    ("attori bravissimi e regia curata",         1),
    ("trama confusa e finale deludente",         0),
]


# ------------------------------------------------------------------
# 4. Classificazione
# ------------------------------------------------------------------

print(f"{'Frase':<45} {'Atteso':<10} {'Stelle':<10} {'Predetto':<10} {'Ok'}")
print("-" * 90)

corretti = 0

for testo, etichetta_vera in frasi:

    inputs = tokenizer(
        testo,
        return_tensors="pt",  # restituisce tensori PyTorch
        truncation=True,      # tronca se supera 512 token
        padding=True          # porta alla stessa lunghezza
    )

    with torch.no_grad():
        # no_grad disabilita il calcolo dei gradienti
        # non stiamo addestrando: risparmia memoria e velocizza
        outputs = model(**inputs)
        # **inputs spacchetta il dizionario come argomenti separati:
        # model(input_ids=..., attention_mask=..., token_type_ids=...)

    indice_predetto = outputs.logits.argmax(dim=-1).item()
    # argmax -> indice del logit più alto tra le 5 classi
    # .item() -> converte tensore scalare in intero Python

    predizione_binaria = stelle_a_binario(indice_predetto)

    corretto = "V" if predizione_binaria == etichetta_vera else "X"
    if predizione_binaria == etichetta_vera:
        corretti += 1

    label_vera  = "positivo" if etichetta_vera == 1 else "negativo"
    label_pred  = "positivo" if predizione_binaria == 1 else "negativo"

    print(f"{testo:<45} {label_vera:<10} {indice_a_stelle(indice_predetto):<10} {label_pred:<10} {corretto}")

print("-" * 90)
print(f"Accuracy: {corretti}/{len(frasi)} = {corretti/len(frasi)*100:.1f}%")


# ------------------------------------------------------------------
# 5. Sotto il cofano — dettaglio su una frase
# ------------------------------------------------------------------

print("\n--- Cosa vede BERT internamente ---")
frase_esempio = "il film è bellissimo davvero"

inputs = tokenizer(frase_esempio, return_tensors="pt")

# vediamo i token prodotti dal tokenizer di BERT
token_ids   = inputs["input_ids"][0].tolist()
token_words = tokenizer.convert_ids_to_tokens(token_ids)
# convert_ids_to_tokens fa il percorso inverso di encode — interi -> stringhe

print(f"Frase:          {frase_esempio}")
print(f"Token:          {token_words}")
print(f"Input IDs:      {token_ids}")
print(f"Attention mask: {inputs['attention_mask'][0].tolist()}")
# cosa notare:
# - [CLS] all'inizio e [SEP] alla fine — aggiunti automaticamente
#   nel nostro tokenizer lo facevamo a mano in data.py
# - i ## indicano subword token — parole rare vengono spezzate
#   es. "bellissimo" -> ["bell", "##issimo"]
#   il nostro SimpleTokenizer non lo fa — tratta le parole intere

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # output_hidden_states=True restituisce gli hidden states di tutti i layer
    # outputs.hidden_states è una tupla di 13 tensori (embedding + 12 layer)

cls_embedding = outputs.hidden_states[-1][:, 0, :]
# hidden_states[-1] = ultimo layer — la rappresentazione più ricca
# [:, 0, :] = token [CLS] — stesso concetto del nostro modello
# differenza: d_model=768 invece di 64 — BERT-base è molto più grande

print(f"\nCLS embedding shape: {cls_embedding.shape}")
# [1, 768] — 768 dimensioni invece delle nostre 64
print(f"Prime 8 dimensioni:  {cls_embedding[0, :8].tolist()}")

