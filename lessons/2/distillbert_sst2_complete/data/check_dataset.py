from datasets import load_dataset
d =load_dataset("dmilush/shieldlm-prompt-injection")
labels = set(d["train"]["label_binary"])
print(labels)
print(d.keys())
print(d["test"].column_names)
print(d["test"][0])
print(f"Esempi: {len(d['test'])}")