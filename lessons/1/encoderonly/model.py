# model.py
import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Positional Encoding — aggiunge informazione di posizione
# ------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)          # dropout per regolarizzazione

        pe = torch.zeros(max_len, d_model)            # matrice vuota [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # indici di posizione [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                             # fattori di scala per le frequenze

        pe[:, 0::2] = torch.sin(position * div_term) # dimensioni pari -> seno
        pe[:, 1::2] = torch.cos(position * div_term) # dimensioni dispari -> coseno

        self.register_buffer('pe', pe.unsqueeze(0))   # salva come buffer non allenabile [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]               # somma la posizione agli embedding
        return self.dropout(x)                        # applica dropout


# ------------------------------------------------------------------
# Encoder Classifier — il modello principale
# ------------------------------------------------------------------

class EncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,       # quante parole conosce il modello
        d_model: int = 64,     # dimensione dei vettori interni
        nhead: int = 2,        # numero di teste di attenzione
        num_layers: int = 2,   # quanti layer di encoder impilare
        dim_feedforward: int = 128,  # dimensione del layer feed-forward interno
        dropout: float = 0.1,  # percentuale di dropout
        num_classes: int = 2,  # classi di output (positivo/negativo)
    ):
        super().__init__()

        # converte ogni token ID in un vettore di dimensione d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        # aggiunge informazione di posizione agli embedding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # definisce un singolo layer di attenzione multi-head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # shape [batch, seq_len, d_model] invece di [seq_len, batch, d_model]
            norm_first=True,    # normalizzazione prima dell'attenzione (più stabile)
            activation="gelu"   # funzione di attivazione (migliore di ReLU per i transformer)
        )

        # impila num_layers layer di attenzione uno sopra l'altro
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # layer finale: da vettore d_model a num_classes logit
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,        # [batch, seq_len] — gli ID dei token
        attention_mask: torch.Tensor,   # [batch, seq_len] — 1=token reale, 0=padding
    ) -> torch.Tensor:                  # [batch, num_classes] — i logit finali

        x = self.embedding(input_ids)               # token IDs -> vettori [batch, seq_len, d_model]
        x = self.pos_encoder(x)                     # aggiunge posizione [batch, seq_len, d_model]
        mask = (attention_mask == 0)                # inverte la maschera: True dove ignorare (padding)
        x = self.encoder(x, src_key_padding_mask=mask)  # passa attraverso i layer di attenzione
        cls_output = x[:, 0, :]                     # prende solo il token [CLS] (primo token) [batch, d_model]
        return self.classifier(cls_output)          # produce i logit finali [batch, num_classes]


# ------------------------------------------------------------------
# Test rapido
# ------------------------------------------------------------------

if __name__ == "__main__":
    model = EncoderClassifier(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        num_classes=2,
    )

    # conta quanti parametri ha il modello
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri allenabili: {n_params:,}")

    # crea dati fittizi per testare il forward pass
    batch_size, seq_len = 4, 16
    input_ids      = torch.randint(0, 100, (batch_size, seq_len))   # IDs casuali
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)  # tutti token reali

    logits = model(input_ids, attention_mask)
    print(f"Input shape:  {input_ids.shape}")   # [4, 16]
    print(f"Output shape: {logits.shape}")      # [4, 2]
    print(f"Output (logits):\n{logits}")