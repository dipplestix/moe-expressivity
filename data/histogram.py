import torch
import numpy as np
from collections import Counter


class HistogramDataset:
    """
    Histogram counting task from Behrens et al. (2025) "Counting in Small Transformers."

    Given a sequence of tokens from alphabet {0, ..., T-1}, predict the frequency
    of each token in the sequence. For example:
        input:  [2, 0, 3, 3, 0, 0]
        output: [3, 1, 2, 2, 3, 3]  (count of each token's value in full sequence)

    Sequences are generated via backward sampling (random partitioning) following
    the original paper, ensuring uniform distribution over partition structures.

    Vocab size = T (alphabet tokens).
    Num classes = L (counts range from 1 to L, mapped to classes 0..L-1).

    Implementation follows SPOC-group/counting-attention (Behrens et al.).
    """

    def __init__(
        self,
        T: int = 32,
        L: int = 10,
        n_train: int = 10000,
        n_test: int = 3000,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.T = T
        self.L = L
        self.vocab_size = T
        self.num_classes = L  # counts 1..L mapped to 0..L-1

        rs = np.random.RandomState(seed)

        train_X, train_y = self._generate(n_train, rs)
        test_X, test_y = self._generate(n_test, rs)

        self.train_inputs = torch.tensor(train_X, dtype=torch.long).to(device)
        self.train_targets = torch.tensor(train_y, dtype=torch.long).to(device)
        self.test_inputs = torch.tensor(test_X, dtype=torch.long).to(device)
        self.test_targets = torch.tensor(test_y, dtype=torch.long).to(device)

    def _random_partition(self, n: int, rs: np.random.RandomState) -> list:
        """Recursively partition n into random positive integer parts.
        Following Behrens et al. backward sampling."""
        if n == 0:
            return []
        if n == 1:
            return [1]
        k = rs.randint(1, n)  # randint is [low, high) in numpy
        return [k] + self._random_partition(n - k, rs)

    def _random_sequence(self, rs: np.random.RandomState) -> np.ndarray:
        """Generate one sequence of length L via backward sampling.

        1. Create a random partition of L
        2. Truncate to at most T groups (merge excess into last group)
        3. Assign a unique random token to each group
        4. Shuffle the sequence
        """
        partition = self._random_partition(self.L, rs)
        partition = np.cumsum(partition)

        # If more groups than tokens, merge excess into last valid group
        if len(partition) > self.T:
            partition = list(partition[:self.T - 1]) + [sum(partition[self.T - 1:])]
            partition = np.array(partition)

        # Assign unique tokens to each group
        tokens = rs.choice(range(self.T), size=len(partition), replace=False)

        # Build sequence
        sequence = np.zeros(self.L, dtype=np.int64)
        sequence[:partition[0]] = tokens[0]
        for i, j, t in zip(partition[:-1], partition[1:], tokens[1:]):
            sequence[i:j] = t

        rs.shuffle(sequence)
        return sequence

    def _hist(self, seq: np.ndarray) -> np.ndarray:
        """Compute per-position histogram label: count of each token in sequence."""
        c = Counter(seq)
        return np.array([c[w] for w in seq])

    def _generate(self, n: int, rs: np.random.RandomState):
        """Generate n sequences with histogram labels."""
        X = np.vstack([self._random_sequence(rs) for _ in range(n)])
        y = np.empty_like(X)
        for i in range(n):
            y[i] = self._hist(X[i]) - 1  # count {1..L} -> class {0..L-1}
        return X, y
