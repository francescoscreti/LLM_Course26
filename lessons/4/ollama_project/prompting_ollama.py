# prompting_ollama.py
# Strategie di prompting: zero-shot, few-shot, chain-of-thought, system prompt
#
# Obiettivo: vedere come cambia la qualità delle risposte
# usando strategie diverse sullo stesso task
import ollama
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL


def chiedi(messages: list, label: str = ""):
    """Invia i messaggi al modello e stampa la risposta."""
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

    response = ollama.chat(model=MODEL, messages=messages)
    risposta = response["message"]["content"]
    print(risposta)
    return risposta


# ------------------------------------------------------------------
# 1. Zero-shot — nessun esempio, solo l'istruzione
# ------------------------------------------------------------------
# Il modello deve capire il task solo dalla descrizione.
# È il punto di partenza — se funziona bene, non serve altro.

chiedi(
    messages=[
        {
            "role": "user",
            "content": "Classifica il sentiment di questa frase come POSITIVO o NEGATIVO:\n'Il film era noioso e lungo.'"
        }
    ],
    label="1. ZERO-SHOT"
)

# ------------------------------------------------------------------
# 2. Few-shot — fornisci esempi prima dell'input reale
# ------------------------------------------------------------------
# Gli esempi guidano il modello sul formato atteso.
# Utile quando il zero-shot produce risposte troppo verbose
# o nel formato sbagliato.

chiedi(
    messages=[
        {
            "role": "user",
            "content": """Classifica il sentiment come POSITIVO o NEGATIVO.

Esempi:
Frase: "Questo ristorante è fantastico!"
Sentiment: POSITIVO

Frase: "Il servizio era pessimo e il cibo freddo."
Sentiment: NEGATIVO

Frase: "Ho adorato ogni momento del film."
Sentiment: POSITIVO

Ora classifica:
Frase: "Il film era noioso e lungo."
Sentiment:"""
        }
    ],
    label="2. FEW-SHOT"
)

# ------------------------------------------------------------------
# 3. Chain-of-thought: chiedi al modello di ragionare step by step
# ------------------------------------------------------------------
# Migliora le performance su task che richiedono ragionamento.
# Il modello "pensa ad alta voce" prima di rispondere.

chiedi(
    messages=[
        {
            "role": "user",
            "content": """Ragiona step by step e poi classifica il sentiment come POSITIVO o NEGATIVO.

Frase: "Il film era noioso e lungo."

Ragionamento:"""
        }
    ],
    label="3. CHAIN-OF-THOUGHT"
)

# ------------------------------------------------------------------
# 4. System prompt — imposta il comportamento del modello
# ------------------------------------------------------------------
# Il system prompt definisce il "ruolo" del modello.
# Viene processato prima di qualsiasi messaggio utente.
# Utile per applicazioni specifiche o per controllare il tono.

chiedi(
    messages=[
        {
            "role": "system",
            "content": "Sei un classificatore di sentiment. Rispondi SEMPRE e SOLO con una parola: POSITIVO o NEGATIVO. Non aggiungere spiegazioni."
        },
        {
            "role": "user",
            "content": "Il film era noioso e lungo."
        }
    ],
    label="4. SYSTEM PROMPT"
)

# ------------------------------------------------------------------
# 5. Confronto diretto — stesso task, strategie diverse
# ------------------------------------------------------------------
# TODO: prova le 4 strategie su questa frase ambigua
# e osserva le differenze nelle risposte
#
# Frase: "Non è stato il peggior film che abbia mai visto."
#
# Domande da discutere:
#   - quale strategia dà la risposta più precisa?
#   - quale strategia è più consistente se la esegui più volte?
#   - quando useresti chain-of-thought invece di few-shot?

print("\n" + "="*50)
print("  TODO — prova le 4 strategie sulla frase ambigua")
print("="*50)
print("Frase: 'Non è stato il peggior film che abbia mai visto.'")
print("Implementa le 4 strategie e confronta i risultati.")