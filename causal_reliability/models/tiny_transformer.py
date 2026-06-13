import torch


class TinyTransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, d_model: int = 32, nhead: int = 4, num_classes: int = 2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.head = torch.nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x.long())
        z = self.encoder(z)
        return self.head(z.mean(dim=1))
