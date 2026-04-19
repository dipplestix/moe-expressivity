# data/

Dataset classes for all three tasks.

## Files

### `modular_addition.py`

`ModularAdditionDataset` — (a+b) mod p, following Nanda et al. (2023).

- Generates all p^2 pairs, deterministic 30/70 train/test split
- Token format: `[a, b, =]` where `=` has token ID p
- Vocab size: p + 1 (default p=113, so vocab=114)
- Labels: (a+b) mod p at position 2

```python
from data.modular_addition import ModularAdditionDataset
ds = ModularAdditionDataset(p=113, train_frac=0.3, seed=42, device="cpu")
# ds.train_inputs:  (3830, 3)  — token sequences
# ds.train_targets: (3830,)    — target values
```

### `histogram.py`

`HistogramDataset` — Token frequency counting, following Behrens et al. (2025).

- Generates sequences via backward sampling (random partitioning)
- Input: sequence of length L from alphabet {0, ..., T-1}
- Target: per-position count of that token in the sequence, mapped to class {0, ..., L-1}
- Vocab size: T (default 32)
- Num classes: L (default 10)

```python
from data.histogram import HistogramDataset
ds = HistogramDataset(T=32, L=10, n_train=10000, n_test=3000, seed=42, device="cpu")
# ds.train_inputs:  (10000, 10) — token sequences
# ds.train_targets: (10000, 10) — per-position count classes
```

Example: input `[2, 0, 3, 3, 0, 0]` → target `[2, 2, 1, 1, 2, 2]` (token 0 appears 3 times → class 2, token 2 appears 1 time → class 0, token 3 appears 2 times → class 1)

### `__init__.py`

Exports: `ModularAdditionDataset`, `HistogramDataset`

## Add-7 Data

Add-7 data is generated on-the-fly in `train.py` (not a separate dataset class). See `train.py:generate_batch()` for the data generation logic. Input format: `[reversed_digits, EOS, reversed_output_digits, EOS]`.
