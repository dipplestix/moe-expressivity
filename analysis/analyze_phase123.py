"""
Analyses for the overnight controls:
  1. Mann-Whitney grokking (n=20) — FFN vs MoE modadd
  2. Phase 2 ablation — add-7 MoE-GLU d=170 (5 seeds)
  3. Phase 3 ablation — histogram GLU/MoE-GLU d=340 (5+5 seeds)

Run: .venv/bin/python analysis/analyze_phase123.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import OneLayerTransformer

DEVICE = "cpu"
P = 113
ORIG_SEEDS = [42, 137, 256, 512, 1024]
EXTRA_SEEDS = list(range(1, 16))
ALL_SEEDS = ORIG_SEEDS + EXTRA_SEEDS  # 20 total


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = ckpt["config"]
    allowed = {
        "model_dim", "num_heads", "ffn_type", "vocab_size", "max_seq_len",
        "use_norm", "is_causal", "tie_embeddings", "activation", "dropout",
        "intermediate_dim", "num_experts", "top_k",
    }
    model_cfg = {k: v for k, v in cfg.items() if k in allowed}
    m = OneLayerTransformer(**model_cfg).to(DEVICE)
    # Resize unembed BEFORE loading (histogram has 10 classes vs 32 vocab)
    if "num_classes" in cfg and cfg["num_classes"] != cfg.get("vocab_size"):
        m.unembed = torch.nn.Linear(cfg["model_dim"], cfg["num_classes"], bias=False).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt


# ============================================================
# 1. Mann-Whitney grokking at n=20
# ============================================================
def analysis_1():
    print("=" * 60)
    print("1. Mann-Whitney grokking (n=20)")
    print("=" * 60)

    CENSOR = 40000  # max epochs

    def grok_epochs(ftype):
        eps, censored = [], 0
        for s in ORIG_SEEDS:
            p = f"checkpoints/modadd_{ftype}_s{s}/modadd_test99.pt"
            if Path(p).exists():
                ck = torch.load(p, weights_only=False, map_location="cpu")
                eps.append(int(ck["epoch"]))
            else:
                eps.append(CENSOR); censored += 1
        for s in EXTRA_SEEDS:
            p = f"checkpoints/modadd_{ftype}_extra_s{s}/modadd_test99.pt"
            if Path(p).exists():
                ck = torch.load(p, weights_only=False, map_location="cpu")
                eps.append(int(ck["epoch"]))
            else:
                eps.append(CENSOR); censored += 1
        return eps, censored

    ffn, ffn_cen = grok_epochs("ffn")
    moe, moe_cen = grok_epochs("moe")
    print(f"FFN n={len(ffn)} ({ffn_cen} censored at {CENSOR}), "
          f"mean={np.mean(ffn):.0f}, median={np.median(ffn):.0f}, std={np.std(ffn):.0f}")
    print(f"  epochs: {sorted(ffn)}")
    print(f"MoE n={len(moe)} ({moe_cen} censored), "
          f"mean={np.mean(moe):.0f}, median={np.median(moe):.0f}, std={np.std(moe):.0f}")
    print(f"  epochs: {sorted(moe)}")

    # Mann-Whitney with censored values as ties at CENSOR (conservative — treats non-grok as worst tied rank)
    u, p = mannwhitneyu(ffn, moe, alternative="greater")
    print(f"\nFull n=20 (censored at {CENSOR}): U={u:.1f}, p={p:.3e}")
    print(f"  FFN median / MoE median = {np.median(ffn)/np.median(moe):.2f}x")

    # Also report uncensored subset
    ffn_g = [e for e in ffn if e < CENSOR]
    moe_g = [e for e in moe if e < CENSOR]
    u2, p2 = mannwhitneyu(ffn_g, moe_g, alternative="greater")
    print(f"Grok-only subset (FFN n={len(ffn_g)}, MoE n={len(moe_g)}): "
          f"U={u2:.1f}, p={p2:.3e}")
    print(f"  FFN median / MoE median = {np.median(ffn_g)/np.median(moe_g):.2f}x")


# ============================================================
# 2. Phase 2 ablation — add-7 MoE-GLU d=170
# ============================================================
def build_add7_data(num_digits=3):
    """Generate all 3-digit add-7 inputs (digits-first reversed)."""
    PAD_TOKEN = 10
    EOS_TOKEN = 11
    xs, ys = [], []
    for n in range(10 ** num_digits):
        inp_digits = [int(d) for d in str(n).zfill(num_digits)][::-1]
        out_val = n + 7
        out_digits = [int(d) for d in str(out_val).zfill(num_digits + 1)][::-1][:num_digits + 1]
        seq = inp_digits + [EOS_TOKEN] + out_digits + [EOS_TOKEN]
        # target = seq shifted by 1 (predict next token)
        target = seq[1:] + [PAD_TOKEN]
        xs.append(seq)
        ys.append(target)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long), len(inp_digits) + 1


def ablation_accuracy_add7(model, x, targets, out_start):
    with torch.no_grad():
        logits = model(x)
    normal = (logits.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()

    orig_attn = model.atn.forward
    model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_na = model(x)
    no_attn = (logits_na.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_nf = model(x)
    no_ffn = (logits_nf.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()
    model.ffn.forward = orig_ffn

    return normal, no_attn, no_ffn


def analysis_2():
    print("\n" + "=" * 60)
    print("2. Phase 2 ablation — add-7 MoE-GLU d=170")
    print("=" * 60)
    x, y, out_start = build_add7_data(num_digits=3)
    results = []
    for s in ORIG_SEEDS:
        p = f"checkpoints/add7_moe_glu_d170_nonorm_s{s}/best_model.pt"
        if not Path(p).exists():
            print(f"  seed={s}: MISSING"); continue
        m, _ = load_model(p)
        normal, na, nf = ablation_accuracy_add7(m, x, y, out_start)
        results.append((s, normal, na, nf))
        print(f"  seed={s}: normal={normal*100:.1f}% no-attn={na*100:.1f}% no-ffn={nf*100:.1f}%")
    if results:
        normals = np.array([r[1] for r in results])
        nas = np.array([r[2] for r in results])
        nfs = np.array([r[3] for r in results])
        print(f"\n  MEAN (n={len(results)}): normal={normals.mean()*100:.1f}% "
              f"no-attn={nas.mean()*100:.1f}±{nas.std()*100:.1f}% "
              f"no-ffn={nfs.mean()*100:.1f}±{nfs.std()*100:.1f}%")
        print(f"  Compare to original add-7 MoE-GLU: no-FFN 40% ± 10% (from paper)")


# ============================================================
# 3. Phase 3 ablation — histogram GLU/MoE-GLU d=340
# ============================================================
def build_histogram_data(T=32, L=10, n=3000, seed=0):
    """Rebuild Behrens-style histogram eval set."""
    from data.histogram import HistogramDataset
    ds = HistogramDataset(T=T, L=L, n_train=1, n_test=n, seed=seed, device="cpu")
    return ds.test_inputs, ds.test_targets


def ablation_accuracy_histogram(model, x, targets):
    """Every position contributes."""
    with torch.no_grad():
        logits = model(x)
    normal = (logits.argmax(-1) == targets).float().mean().item()

    orig_attn = model.atn.forward
    model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_na = model(x)
    no_attn = (logits_na.argmax(-1) == targets).float().mean().item()
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_nf = model(x)
    no_ffn = (logits_nf.argmax(-1) == targets).float().mean().item()
    model.ffn.forward = orig_ffn

    return normal, no_attn, no_ffn


def analysis_3():
    print("\n" + "=" * 60)
    print("3. Phase 3 ablation — histogram param-matched (d=340)")
    print("=" * 60)

    # Build eval set once (different seed per checkpoint would be fine; use seed=0 for consistency)
    x, y = build_histogram_data(seed=0)

    for variant, dir_prefix in [("GLU d=340", "hist_glu_d340_s"), ("MoE-GLU d=340", "hist_moe_glu_d340_s")]:
        print(f"\n-- {variant} --")
        results = []
        for s in ORIG_SEEDS:
            p = f"checkpoints/{dir_prefix}{s}/hist_best.pt"
            if not Path(p).exists():
                print(f"  seed={s}: MISSING"); continue
            m, _ = load_model(p)
            normal, na, nf = ablation_accuracy_histogram(m, x, y)
            results.append((s, normal, na, nf))
            print(f"  seed={s}: normal={normal*100:.1f}% no-attn={na*100:.1f}% no-ffn={nf*100:.1f}%")
        if results:
            nrm = np.array([r[1] for r in results])
            nas = np.array([r[2] for r in results])
            nfs = np.array([r[3] for r in results])
            print(f"  MEAN (n={len(results)}): normal={nrm.mean()*100:.1f}% "
                  f"no-attn={nas.mean()*100:.1f}±{nas.std()*100:.1f}% "
                  f"no-ffn={nfs.mean()*100:.1f}±{nfs.std()*100:.1f}%")


if __name__ == "__main__":
    analysis_1()
    analysis_2()
    analysis_3()
