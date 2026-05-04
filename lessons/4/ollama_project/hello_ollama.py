# hello_ollama.py
# Prima connessione a Ollama: chiamata base
#
# Prima di eseguire:
#   1. ollama serve (se non parte automaticamente)
#   2. ollama pull gemma2:2b (o llama3.2:3b)
#   3. pip install ollama

import ollama

# ------------------------------------------------------------------
# 0. Configurazione — cambia il modello qui
# ------------------------------------------------------------------

MODEL = "gemma2:2b"
# oppure: MODEL = "llama3.2:3b"
# oppure usa file config.py per centralizzare la configurazione
# ------------------------------------------------------------------
# 1. Chiamata base — una singola domanda
# ------------------------------------------------------------------

print("=" * 50)
print(f"Modello: {MODEL}")
print("=" * 50)

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Cos'è un Large Language Model? Rispondi in 3 frasi."
        }
    ]
)

print("\nRisposta:")
print(response["message"]["content"])

# ------------------------------------------------------------------
# 2. Streaming — mostra la risposta token per token
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("Stesso prompt con streaming:")
print("=" * 50 + "\n")

stream = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Cos'è un Large Language Model? Rispondi in 3 frasi."
        }
    ],
    stream=True,
    # stream=True restituisce un generatore — ogni elemento è un token
    # invece di aspettare tutta la risposta, la stampa man mano che arriva
    # è come vedere il modello "scrivere" in tempo reale
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
    # end=""  → non va a capo dopo ogni token
    # flush=True → stampa subito senza aspettare il buffer

print("\n")

# ------------------------------------------------------------------
# 3. Informazioni sul modello
# ------------------------------------------------------------------

print("=" * 50)
print("Modelli disponibili localmente:")
print("=" * 50)

modelli = ollama.list()
for m in modelli["models"]:
    size_gb = m["size"] / 1e9
    print(f"  {m['model']:<30} {size_gb:.1f} GB")