import torch


class ModularAdditionDataset:
    """
    Full dataset for (a + b) mod p.

    Generates all p^2 pairs, splits into 30% train / 70% test with a seeded shuffle.
    Token format: [a, b, =] where = token has ID p. Vocab size = p + 1.
    Labels: (a + b) mod p at position 2 (the = position).
    """

    def __init__(self, p: int = 113, train_frac: float = 0.3, seed: int = 42, device: str = "cpu"):
        self.p = p
        self.vocab_size = p + 1  # tokens 0..p-1 for digits, p for '='
        self.eq_token = p

        # Generate all p^2 pairs
        a = torch.arange(p).repeat_interleave(p)
        b = torch.arange(p).repeat(p)
        targets = (a + b) % p

        # Token sequences: [a, b, =]
        eq = torch.full_like(a, self.eq_token)
        sequences = torch.stack([a, b, eq], dim=1)  # (p^2, 3)

        # Deterministic shuffle and split
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(sequences), generator=gen)
        sequences = sequences[perm]
        targets = targets[perm]

        n_train = int(len(sequences) * train_frac)

        self.train_inputs = sequences[:n_train].to(device)
        self.train_targets = targets[:n_train].to(device)
        self.test_inputs = sequences[n_train:].to(device)
        self.test_targets = targets[n_train:].to(device)
