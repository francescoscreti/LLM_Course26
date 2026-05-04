# chat_interattiva.py
# Chat interattiva senza memoria
# Ogni messaggio è indipendente: il modello non ricorda nulla

import ollama
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL



print("=" * 50)
print("  Chat senza memoria")
print("  Ogni messaggio è indipendente")
print("  digita 'exit' per uscire")
print("=" * 50 + "\n")

while True:
    user_input = input("Tu: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Arrivederci!")
        break

    print("\nAssistente: ", end="", flush=True)

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": user_input}],
        # nota: passiamo solo il messaggio corrente
        # nessuna history — il modello non sa nulla dei turni precedenti
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

    print()