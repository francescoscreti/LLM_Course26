import ollama

MODEL = "gemma2:2b" # oppure: MODEL = "llama3.2:3b"

response = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Rispondi SOLO in maiuscolo."},
        {"role": "user",   "content": "Come stai?"}
    ]
)
print(response["message"]["content"])
#se risponde in maiuscolo: system prompt supportato
#se risponde normalmente: non supportato o ignorato