# dataset_qa.py
# Dataset di domande e risposte per testare le strategie di prompting


QA_DATASET = [
    # geografia
    {"question": "What is the capital of France?",          "answer": "Paris"},
    {"question": "What is the capital of Japan?",           "answer": "Tokyo"},
    {"question": "What is the longest river in the world?", "answer": "Nile"},
    {"question": "What is the highest mountain in the world?", "answer": "Mount Everest"},
    {"question": "What is the capital of Australia?",       "answer": "Canberra"},

    # scienza
    {"question": "What is the chemical symbol for water?",  "answer": "H2O"},
    {"question": "How many planets are in the solar system?", "answer": "8"},
    {"question": "What is the speed of light in km/s?",     "answer": "300000"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "CO2"},
    {"question": "What is the powerhouse of the cell?",     "answer": "mitochondria"},

    # storia
    {"question": "In what year did World War II end?",      "answer": "1945"},
    {"question": "Who painted the Mona Lisa?",              "answer": "Leonardo da Vinci"},
    {"question": "In what year did man first land on the moon?", "answer": "1969"},
    {"question": "Who wrote Romeo and Juliet?",             "answer": "Shakespeare"},
    {"question": "What year was the Eiffel Tower built?",   "answer": "1889"},

    # matematica e logica
    {"question": "What is the square root of 144?",         "answer": "12"},
    {"question": "What is 15% of 200?",                     "answer": "30"},
    {"question": "How many sides does a hexagon have?",     "answer": "6"},
    {"question": "What is the next prime number after 7?",  "answer": "11"},
    {"question": "What is 2 to the power of 10?",           "answer": "1024"},
]