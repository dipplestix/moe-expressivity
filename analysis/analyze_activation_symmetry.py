"""
Comprehensive activation-symmetry analysis.

Covers:
  add-7   main     {FFN, GLU, MoE, MoE-GLU} x {SiLU (existing), GELU (new Tier 1)}
  add-7   h=170    {GLU, MoE-GLU}           x {SiLU, GELU}
  hist    main     {FFN, GLU, MoE, MoE-GLU} x {GELU (existing), SiLU (new Tier 2)}
  hist    h=340    {GLU, MoE-GLU}           x {GELU (existing), SiLU (new Tier 3)}

Loader fix: when `config.activation` is None (pre-fix train.py or pre-flag training),
falls back to `args.activation` if present; otherwise assumes 'silu' (class default).

Run: /tmp/moe-venv/bin/python analysis/analyze_activation_symmetry.py
"""
import sys, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path("<PATH_TO_REPO>")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import OneLayerTransformer

DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]
PAD_TOKEN = 10
EOS_TOKEN = 11


def _infer_intermediate_dim(state_dict, ffn_type, num_experts):
    """Derive actual intermediate_dim from saved weight shapes (robust to the train.py bug
    that incorrectly stored model_dim*4 for GLU when --intermediate_dim wasn't passed)."""
    if ffn_type == "ffn":
        w = state_dict.get("ffn.up_proj.weight")
        return w.shape[0] if w is not None else None
    if ffn_type == "glu":
        w = state_dict.get("ffn.up_proj.weight")
        return w.shape[0] if w is not None else None
    if ffn_type == "moe":
        w = state_dict.get("ffn.experts.0.up_proj.weight")
        return w.shape[0] * num_experts if w is not None else None
    if ffn_type == "moe_glu":
        w = state_dict.get("ffn.experts.0.up_proj.weight")
        return w.shape[0] * num_experts if w is not None else None
    return None


def load_model(path):
    """Load model, resolving activation correctly from config or args fallback.
    Also infers intermediate_dim from state_dict shapes (defensively)."""
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = dict(ckpt["config"])
    args = ckpt.get("args", {}) or {}
    act = cfg.get("activation")
    if act is None:
        act = args.get("activation")
    if act is None:
        act = "silu"  # pre --activation flag: class default
    cfg["activation"] = act
    # Infer actual intermediate_dim from weights to sidestep the config save bug
    inferred = _infer_intermediate_dim(ckpt["model_state_dict"], cfg.get("ffn_type"),
                                       cfg.get("num_experts", 4))
    if inferred is not None and inferred != cfg.get("intermediate_dim"):
        cfg["intermediate_dim"] = inferred
    allowed = {"model_dim","num_heads","ffn_type","vocab_size","max_seq_len",
               "use_norm","is_causal","tie_embeddings","activation","dropout",
               "intermediate_dim","num_experts","top_k"}
    mcfg = {k: v for k, v in cfg.items() if k in allowed}
    m = OneLayerTransformer(**mcfg).to(DEVICE)
    if "num_classes" in cfg and cfg["num_classes"] != cfg.get("vocab_size"):
        m.unembed = torch.nn.Linear(cfg["model_dim"], cfg["num_classes"], bias=False).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt, act


def build_add7_data(num_digits=3):
    xs, ys = [], []
    for n in range(10 ** num_digits):
        inp = [int(d) for d in str(n).zfill(num_digits)][::-1]
        out_val = n + 7
        out = [int(d) for d in str(out_val).zfill(num_digits + 1)][::-1][:num_digits + 1]
        seq = inp + [EOS_TOKEN] + out + [EOS_TOKEN]
        target = seq[1:] + [PAD_TOKEN]
        xs.append(seq); ys.append(target)
    # out_start = num_digits points at the predicted o0; downstream code averages
    # the next (num_digits + 1) positions = o0..o3, the four real output digits.
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long), len(inp)


def build_hist_data(T=32, L=10, n=3000, seed=0):
    from data.histogram import HistogramDataset
    ds = HistogramDataset(T=T, L=L, n_train=1, n_test=n, seed=seed, device="cpu")
    return ds.test_inputs, ds.test_targets


def build_modadd_data(p=113):
    """All p^2 (a, b) pairs with target (a+b) mod p."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)  # separator token
    inputs = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p
    return inputs, targets


def ablate_modadd(m, x, y):
    """Ablation on modadd: evaluate final-position output."""
    with torch.no_grad():
        logits = m(x)
    normal = (logits[:, 2, :].argmax(dim=-1) == y).float().mean().item()

    orig = m.atn.forward
    m.atn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_attn = (l[:, 2, :].argmax(dim=-1) == y).float().mean().item()
    m.atn.forward = orig

    orig = m.ffn.forward
    m.ffn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_ffn = (l[:, 2, :].argmax(dim=-1) == y).float().mean().item()
    m.ffn.forward = orig
    return normal, no_attn, no_ffn


def ablate_add7(m, x, y, out_start):
    out_end = out_start + 4  # average over o0..o3 only (exclude trailing EOS / PAD positions)
    with torch.no_grad():
        logits = m(x)
    normal = (logits.argmax(-1)[:, out_start:out_end] == y[:, out_start:out_end]).float().mean().item()
    orig = m.atn.forward
    m.atn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_attn = (l.argmax(-1)[:, out_start:out_end] == y[:, out_start:out_end]).float().mean().item()
    m.atn.forward = orig
    orig = m.ffn.forward
    m.ffn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_ffn = (l.argmax(-1)[:, out_start:out_end] == y[:, out_start:out_end]).float().mean().item()
    m.ffn.forward = orig
    return normal, no_attn, no_ffn


def ablate_hist(m, x, y):
    with torch.no_grad():
        logits = m(x)
    normal = (logits.argmax(-1) == y).float().mean().item()
    orig = m.atn.forward
    m.atn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_attn = (l.argmax(-1) == y).float().mean().item()
    m.atn.forward = orig
    orig = m.ffn.forward
    m.ffn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_ffn = (l.argmax(-1) == y).float().mean().item()
    m.ffn.forward = orig
    return normal, no_attn, no_ffn


def run_group(task, label, dir_templates, ckpt_file, ablate_fn, x, y, extra=None):
    """dir_templates: list of (activation_label, dir_prefix_template) where template has {seed}."""
    print(f"\n==== {task}: {label} ====")
    results = {}
    for act_label, tmpl in dir_templates:
        rows = []
        acts_used = []
        for s in SEEDS:
            d = tmpl.format(seed=s)
            p = ROOT / "checkpoints" / d / ckpt_file
            if not p.exists():
                continue
            try:
                m, _, act_used = load_model(p)
            except Exception as e:
                print(f"  [load-err] {d}: {e}")
                continue
            acts_used.append(act_used)
            if extra is not None:
                n, na, nf = ablate_fn(m, x, y, extra)
            else:
                n, na, nf = ablate_fn(m, x, y)
            rows.append((s, n, na, nf))
        if rows:
            nrm = np.array([r[1] for r in rows])
            nas = np.array([r[2] for r in rows])
            nfs = np.array([r[3] for r in rows])
            activated = set(acts_used)
            act_note = ",".join(sorted(activated)) if activated else "?"
            print(f"  [{act_label:12s}] n={len(rows)}  (loaded as {act_note})  "
                  f"normal={nrm.mean()*100:6.2f}%  "
                  f"no-attn={nas.mean()*100:6.2f}±{nas.std()*100:4.2f}%  "
                  f"no-ffn={nfs.mean()*100:6.2f}±{nfs.std()*100:4.2f}%")
            results[act_label] = {
                "n": len(rows), "normal": nrm.mean(),
                "no_attn_mean": nas.mean(), "no_attn_std": nas.std(),
                "no_ffn_mean": nfs.mean(),  "no_ffn_std": nfs.std(),
                "act_used": act_note,
            }
        else:
            print(f"  [{act_label:12s}] NO CHECKPOINTS FOUND  (tried {tmpl.format(seed='*')})")
    return results


def main():
    all_results = {}

    # ---------- add-7 ----------
    x7, y7, out_start = build_add7_data(num_digits=3)
    print(f"\n####### add-7 (out_start={out_start}) #######")

    add7_cases = [
        ("main FFN",     [("SiLU (existing)", "add7_ffn_nonorm_s{seed}"),
                          ("GELU (Tier 1)",   "add7_ffn_gelu_nonorm_s{seed}")]),
        ("main GLU",     [("SiLU (existing)", "add7_glu_nonorm_s{seed}"),
                          ("GELU (Tier 1)",   "add7_glu_gelu_nonorm_s{seed}")]),
        ("main MoE",     [("SiLU (existing)", "add7_moe_nonorm_s{seed}"),
                          ("GELU (Tier 1)",   "add7_moe_gelu_nonorm_s{seed}")]),
        ("main MoE-GLU", [("SiLU (existing)", "add7_moe_glu_nonorm_s{seed}"),
                          ("GELU (Tier 1)",   "add7_moe_glu_gelu_nonorm_s{seed}")]),
        ("h=170 GLU",    [("SiLU",            "add7_glu_d170_silu_nonorm_s{seed}"),
                          ("GELU",            "add7_glu_d170_gelu_nonorm_s{seed}")]),
        ("h=170 MoE-GLU",[("SiLU (queued)",   "add7_moe_glu_d170_silu_nonorm_s{seed}"),
                          ("GELU (P2+PB)",    "add7_moe_glu_d170_gelu_nonorm_s{seed}")]),
    ]
    for label, templates in add7_cases:
        all_results[("add7", label)] = run_group("add-7", label, templates,
                                                  "best_model.pt", ablate_add7, x7, y7, out_start)

    # ---------- modular addition ----------
    try:
        xm, ym = build_modadd_data(p=113)
        print(f"\n####### modular addition (n={xm.shape[0]} pairs) #######")
        modadd_cases = [
            ("main FFN",     [("GELU (existing)", "modadd_ffn_s{seed}"),
                              ("SiLU (new)",      "modadd_ffn_silu_s{seed}")]),
            ("main GLU",     [("GELU (existing)", "modadd_glu_s{seed}"),
                              ("SiLU (existing)", "modadd_glu_silu_s{seed}")]),
            ("main MoE",     [("GELU (existing)", "modadd_moe_s{seed}"),
                              ("SiLU (new)",      "modadd_moe_silu_s{seed}")]),
            ("main MoE-GLU", [("GELU (existing)", "modadd_moe_glu_s{seed}"),
                              ("SiLU (existing)", "modadd_moe_glu_silu_s{seed}")]),
        ]
        for label, templates in modadd_cases:
            # modadd checkpoints: prefer modadd_test99.pt (grokked), fall back to modadd_best.pt
            rows_per_label = {}
            for act_label, tmpl in templates:
                rows = []
                acts = []
                for s in SEEDS:
                    d = ROOT / "checkpoints" / tmpl.format(seed=s)
                    ck_file = None
                    for fn in ["modadd_test99.pt", "modadd_best.pt"]:
                        if (d / fn).exists():
                            ck_file = d / fn; break
                    if ck_file is None:
                        continue
                    try:
                        m, _, act_used = load_model(ck_file)
                    except Exception as e:
                        print(f"  [load-err] {d.name}/{ck_file.name}: {e}")
                        continue
                    acts.append(act_used)
                    n, na, nf = ablate_modadd(m, xm, ym)
                    rows.append((s, n, na, nf))
                rows_per_label[act_label] = rows
                if rows:
                    nrm = np.array([r[1] for r in rows])
                    nas = np.array([r[2] for r in rows])
                    nfs = np.array([r[3] for r in rows])
                    print(f"  [{act_label:16s}] n={len(rows)}  (loaded as {','.join(sorted(set(acts)))})  "
                          f"normal={nrm.mean()*100:6.2f}%  "
                          f"no-attn={nas.mean()*100:5.2f}±{nas.std()*100:4.2f}%  "
                          f"no-ffn={nfs.mean()*100:5.2f}±{nfs.std()*100:4.2f}%")
                    all_results[("modadd", label)] = all_results.get(("modadd", label), {})
                    all_results[("modadd", label)][act_label] = {
                        "n": len(rows), "normal": nrm.mean(),
                        "no_attn_mean": nas.mean(), "no_attn_std": nas.std(),
                        "no_ffn_mean": nfs.mean(), "no_ffn_std": nfs.std(),
                        "act_used": ",".join(sorted(set(acts))),
                    }
                else:
                    print(f"  [{act_label:16s}] no checkpoints found")
    except Exception as e:
        print(f"\n[modadd] SKIP: {e}")

    # ---------- histogram ----------
    try:
        xh, yh = build_hist_data(seed=0)
        print(f"\n####### histogram (n={xh.shape[0]}) #######")
        hist_cases = [
            ("main FFN",     [("GELU (existing)", "hist_ffn_s{seed}"),
                              ("SiLU (Tier 2)",   "hist_ffn_silu_s{seed}")]),
            ("main GLU",     [("GELU (existing)", "hist_glu_s{seed}"),
                              ("SiLU (Tier 2)",   "hist_glu_silu_s{seed}")]),
            ("main MoE",     [("GELU (existing)", "hist_moe_s{seed}"),
                              ("SiLU (Tier 2)",   "hist_moe_silu_s{seed}")]),
            ("main MoE-GLU", [("GELU (existing)", "hist_moe_glu_s{seed}"),
                              ("SiLU (Tier 2)",   "hist_moe_glu_silu_s{seed}")]),
            ("d=340 GLU",    [("GELU (existing)", "hist_glu_d340_s{seed}"),
                              ("SiLU (Tier 3)",   "hist_glu_d340_silu_s{seed}")]),
            ("d=340 MoE-GLU",[("GELU (existing)", "hist_moe_glu_d340_s{seed}"),
                              ("SiLU (Tier 3)",   "hist_moe_glu_d340_silu_s{seed}")]),
        ]
        for label, templates in hist_cases:
            all_results[("hist", label)] = run_group("hist", label, templates,
                                                      "hist_best.pt", ablate_hist, xh, yh)
    except Exception as e:
        print(f"\n[hist] SKIP: {e}")

    # ---------- SUMMARY ----------
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE  —  activation symmetry across tasks and param-matching")
    print("=" * 100)
    print(f"{'task':6s} {'condition':15s} {'activation':18s} {'n':>2s}  {'normal':>8s}  {'no-attn':>14s}  {'no-ffn':>14s}")
    print("-" * 100)
    for (task, cond), acts in sorted(all_results.items()):
        for act_label, r in acts.items():
            print(f"{task:6s} {cond:15s} {act_label:18s} {r['n']:2d}  "
                  f"{r['normal']*100:7.2f}%  "
                  f"{r['no_attn_mean']*100:6.2f}±{r['no_attn_std']*100:<5.2f}%  "
                  f"{r['no_ffn_mean']*100:6.2f}±{r['no_ffn_std']*100:<5.2f}%")


if __name__ == "__main__":
    main()
