# OLLAMA
https://www.geeksforgeeks.org/artificial-intelligence/what-is-ollama/ 

Ollama è uno strumento open source che permette di eseguire LLMs direttamente sul proprio computer, senza bisogno di connessione internet o API esterne.

Funziona come un server locale: scarichi il modello una volta sola, poi lo chiami dal codice Python esattamente come faresti con un'API remota.

```
Il tuo codice Python
↓
ollama (client)
↓
Ollama server (locale)
↓
modello (es. gemma2:2b)
↓
risposta
```
Ollama server: scaricato da ollama.com/download
                 fa girare i modelli in locale
                 si avvia con: ollama serve

ollama (Python): pip install ollama
                  è il client che parla con il server
                  usato nel codice Python

---

## Installazione

### 1. Installa Ollama

Scarica e installa Ollama dal sito ufficiale: https://ollama.com/


Scegli la versione per il tuo sistema operativo (Mac, Windows, Linux).

### 2. Scarica un modello

Apri il terminale e scegli uno dei due modelli consigliati per questa lezione:

```bash
# modello leggero: consigliato se hai meno di 8GB di RAM
ollama pull gemma2:2b

# modello più capace — consigliato se hai 8GB+ di RAM
ollama pull llama3.2:3b
```

Il download richiede qualche minuto — i modelli pesano circa 1.5-2GB.

### 3. Verifica che funzioni

```bash
ollama run gemma2:2b "Ciao, come stai?"
```

Se vedi una risposta, Ollama è installato correttamente.

### 4. Installa il client Python

```bash
pip install ollama
```

### 5. Avvia il server (se necessario)

Su Mac e Windows Ollama si avvia automaticamente dopo l'installazione.
Su Linux potrebbe essere necessario avviarlo manualmente:

```bash
ollama serve
```

---

## Modelli disponibili

| Modello | Dimensione | RAM consigliata | Velocità |
|---------|-----------|-----------------|----------|
| gemma2:2b | ~1.5GB | 4GB+ | molto veloce |
| llama3.2:3b | ~2GB | 8GB+ | veloce |

Per vedere tutti i modelli disponibili: [ollama.com/library](https://ollama.com/library)

# API e richieste http
## Richieste HTTP

HTTP è il protocollo che permette la comunicazione tra client e server.
Esistono quattro metodi principali:

```
GET     → recupera dati      "dammi questa risorsa"
POST    → invia dati         "elabora questo input e restituisci un risultato"
PUT     → aggiorna dati      "sostituisci questa risorsa"
DELETE  → elimina dati       "cancella questa risorsa"
```
Nel contesto degli LLM si usa quasi sempre **POST** — stai inviando
testo al server e ricevi una risposta elaborata.

### Esempio con Ollama

Ollama espone una API REST locale sulla porta `11434`.
Puoi chiamarla direttamente con `requests`:

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "gemma2:2b",
        "messages": [{"role": "user", "content": "Ciao!"}]
    }
)
print(response.json())
```

Oppure usando il client Python di Ollama che gestisce le richieste
HTTP internamente:

```python
import ollama

# dietro le quinte fa una POST a localhost:11434/api/chat
response = ollama.chat(
    model="gemma2:2b",
    messages=[{"role": "user", "content": "Ciao!"}]
)
print(response["message"]["content"])
```

La seconda versione è più semplice e leggibile: useremo questa durante la lezione.