# api_requests.py
# Chiamata API esplicita a Ollama con requests
#
# Obiettivo: capire cosa succede "sotto il cofano" quando
# usiamo ollama.chat() — è semplicemente una richiesta HTTP POST

import requests
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL
# ------------------------------------------------------------------
# Configurazione
# ------------------------------------------------------------------

BASE_URL = "http://localhost:11434"
# Ollama gira in locale sulla porta 11434
# è un server HTTP come qualsiasi altro



# ------------------------------------------------------------------
# 1. Verifica che Ollama sia attivo
# ------------------------------------------------------------------

print("=" * 50)
print("  1. Verifica connessione")
print("=" * 50)

try:
    response = requests.get(f"{BASE_URL}")
    print(f"Status code: {response.status_code}")
    print(f"Risposta:    {response.text}")
    # se Ollama è attivo risponde con "Ollama is running"
except requests.exceptions.ConnectionError:
    print("Ollama non raggiungibile — esegui: ollama serve")
    exit(1)

# ------------------------------------------------------------------
# 2. Lista modelli disponibili — GET
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("  2. Lista modelli — GET /api/tags")
print("=" * 50)

response = requests.get(f"{BASE_URL}/api/tags")
# GET perché stiamo recuperando una risorsa esistente
# non stiamo inviando dati

modelli = response.json()["models"]
for m in modelli:
    print(f"  {m['name']}")

# ------------------------------------------------------------------
# 3. Chiamata base — POST senza streaming
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("  3. Chiamata base — POST /api/chat")
print("=" * 50)

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Cos'è un LLM in 2 frasi?"}
    ],
    "stream": False
    # stream=False -> aspetta tutta la risposta prima di restituirla
}

response = requests.post(
    f"{BASE_URL}/api/chat",
    json=payload
    # json= serializza automaticamente il dizionario in JSON
    # e imposta Content-Type: application/json nell'header
)

print(f"Status code: {response.status_code}")
# 200 = OK, 404 = modello non trovato, 500 = errore server

data = response.json()
print(f"Risposta: {data['message']['content']}")

# ------------------------------------------------------------------
# 4. Chiamata con streaming — POST con stream=True
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("  4. Streaming — POST /api/chat con stream=True")
print("=" * 50)

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Cos'è un LLM in 2 frasi?"}
    ],
    "stream": True
}

response = requests.post(
    f"{BASE_URL}/api/chat",
    json=payload,
    stream=True
    # stream=True in requests significa "non scaricare tutto subito"
    # leggi il body pezzo per pezzo con iter_lines()
)

print("Risposta: ", end="", flush=True)

for line in response.iter_lines():
    # iter_lines() legge il body riga per riga
    # ogni riga è un JSON con un token
    if line:
        chunk = json.loads(line)
        token = chunk["message"]["content"]
        print(token, end="", flush=True)

        if chunk.get("done"):
            # done=True indica che la generazione è finita
            break

print("\n")

# ------------------------------------------------------------------
# 5. Confronto — requests vs ollama.chat()
# ------------------------------------------------------------------

print("=" * 50)
print("  5. Confronto requests vs ollama.chat()")
print("=" * 50)

print("""
requests.post("/api/chat", json=payload)
  -> chiamata HTTP esplicita
  -> controllo totale sulla richiesta
  -> devi gestire JSON manualmente

ollama.chat(model=MODEL, messages=[...])
  -> wrapper comodo intorno a requests
  -> stesso risultato, meno codice
  -> usato nei file precedenti

Sono equivalenti — ollama.chat() fa esattamente
quello che hai scritto nel punto 3.
""")