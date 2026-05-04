# TRACCE PROGETTI
Modalità di consegna: creare una cartella apposita divisa in moduli sul vostro repository. 

```
project_name/
├── README.md          descrizione del progetto
├── fine_tuning/       Parte 1
└── ollama_pipeline/   Parte 2
```

## PARTE 1
### OPZIONE A: encoder-only
Fine-tuning di distilBERT (o modelli simili) su un dataset a scelta per un task di classificazione. Riportare e commentare i risultati ottenuti con diverse opzioni per gli iperparametri.

### OPZIONE B: encoder-decoder
Fine-tuning di T5 o BART come visto a lezione su task di summarization o traduzione (chi vuole può sperimentare Question-answering: generare una risposta data una domanda, dataset: https://huggingface.co/datasets/crux82/squad_it).
Riportare e commentare i risultati ottenuti con diverse opzioni per gli iperparametri.

Nota su QA: T5 gestisce QA con i seguenti prefissi
```
"question: What is the capital of Italy? context: Italy is a country in Europe. Its capital is Rome."
-> "Rome"
```

### OPZIONE C: encoder-only from scratch
Utilizzare il codice costruito insieme in lessons/1/encoderonly_solutions e provare ad addestrare un encoder-only da zero su un task di classificazione. Si possono usare i dataset affrontati durante il corso.

## PARTE 2

Implementare una pipeline con Ollama per risolvere un task a scelta includendo almeno due delle seguenti strategie di prompting:
- Zero-shot
- Few-shot
- Chain-of-thought
- System prompt personalizzato

Alcuni esempi:
- Classificatore di testi con few-shot prompting
- Generatore di riassunti con chain-of-thought
- Chatbot tematico con system prompt (specifico su un dominio)
- Comparatore di strategie: zero-shot vs few-shot sullo stesso input
- Pipeline RAG su documenti (bonus)

# Cosa deve contenere il file README

## Progetto finale 
Autore: Nome e cognome

### Parte 1 — Fine-tuning
- Modello scelto e motivazione
- Dataset scelto e motivazione
- Risultati ottenuti (metriche)
- Difficoltà incontrate

### Parte 2 — Pipeline Ollama
- Task scelto e motivazione
- Strategie di prompting usate
- Esempio di input/output
- Considerazioni sui risultati

# Presentazione finale
14 maggio: 10 minuti per gruppo. 

## Struttura della presentazione

### 1. Introduzione al progetto
- Quale task avete scelto e perché
- Quale dataset avete usato e perché
- Quale modello avete scelto e perché 

### 2. Approccio e sviluppo
- Come avete affrontato il problema
- Quali difficoltà avete incontrato ( teoriche, pratiche)
- Come le avete risolte
- Se avete usato LLM (Claude, ChatGPT, Copilot ecc.) come supporto:
  - per cosa li avete usati (debug, spiegazioni, codice, ricerca dataset...)
  - cosa ha funzionato e cosa no
  - c'è qualcosa che il modello vi ha spiegato in modo sbagliato o fuorviante?

### 3. Risultati 
- Plot delle loss e delle metriche di valutazione
- Esempi concreti di input/output del modello fine-tuned
- Schema della pipeline di ollama
- Confronto tra le strategie di prompting usate




