import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditionFFN(nn.Module):
    """Two-layer feed-forward network for addition."""

    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.ReLU(),
        )
        self.out_proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.ffn(x)
        logits = self.out_proj(hidden)
        return F.softmax(logits, dim=-1)


def generate_data(batch_size: int = 32):
    """Generate random single-digit addition data."""
    a = torch.randint(0, 10, (batch_size,))
    b = torch.randint(0, 10, (batch_size,))
    x = torch.stack([a, b], dim=1)
    y = a + b
    return x, y


def train(model: AdditionFFN, embedding: nn.Embedding, epochs: int = 2000):
    """Train the model using the provided embedding."""
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        x, y = generate_data()
        x_emb = embedding(x).mean(dim=1)
        pred = model(x_emb)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    model_dim = 32
    vocab_size = 19  # range of sums 0-18
    model = AdditionFFN(model_dim, vocab_size)
    embedding = nn.Embedding(10, model_dim)
    train(model, embedding, epochs=1000)
    # Test on a few examples
    test_inputs, targets = generate_data(5)
    with torch.no_grad():
        preds = model(embedding(test_inputs).mean(dim=1)).argmax(dim=-1)
    for pair, pred, target in zip(test_inputs.tolist(), preds.tolist(), targets.tolist()):
        print(f"{pair[0]} + {pair[1]} = {pred} (expected {target})")
