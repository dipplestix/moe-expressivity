"""Generate grouped bar chart comparing GELU vs SiLU across tasks and architectures."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

CKPT = "/Users/gabesmithline/Desktop/moe-expressivity/checkpoints"
SEEDS = [42, 137, 256, 512, 1024]

conditions = [
    ("GLU",     f"{CKPT}/add7_glu_nonorm_s{{seed}}/best_model.pt",
                f"{CKPT}/add7_glu_silu_nonorm_s{{seed}}/best_model.pt",      "accuracy"),
    ("MoE-GLU", f"{CKPT}/add7_moe_glu_nonorm_s{{seed}}/best_model.pt",
                f"{CKPT}/add7_moe_glu_silu_nonorm_s{{seed}}/best_model.pt",  "accuracy"),
    ("GLU",     f"{CKPT}/modadd_glu_s{{seed}}/modadd_best.pt",
                f"{CKPT}/modadd_glu_silu_s{{seed}}/modadd_best.pt",          "test_acc"),
    ("MoE-GLU", f"{CKPT}/modadd_moe_glu_s{{seed}}/modadd_best.pt",
                f"{CKPT}/modadd_moe_glu_silu_s{{seed}}/modadd_best.pt",      "test_acc"),
    ("GLU",     f"{CKPT}/hist_glu_s{{seed}}/hist_best.pt",
                f"{CKPT}/hist_glu_silu_s{{seed}}/hist_best.pt",              "test_acc"),
    ("MoE-GLU", f"{CKPT}/hist_moe_glu_s{{seed}}/hist_best.pt",
                f"{CKPT}/hist_moe_glu_silu_s{{seed}}/hist_best.pt",          "test_acc"),
]

def load_accs(template, acc_key):
    accs = []
    for seed in SEEDS:
        path = template.format(seed=seed)
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            val = ckpt.get(acc_key)
            if val is None:
                val = ckpt.get('accuracy') or ckpt.get('test_acc')
            accs.append(float(val) * 100)
        except Exception as e:
            print(f"WARNING: Could not load {path}: {e}")
    return np.array(accs)

# Collect data
gelu_data, silu_data, labels = [], [], []
for label, gelu_tmpl, silu_tmpl, acc_key in conditions:
    labels.append(label)
    gelu_data.append(load_accs(gelu_tmpl, acc_key))
    silu_data.append(load_accs(silu_tmpl, acc_key))

n_groups = len(labels)

# Print summary
print("\n=== GELU vs SiLU Comparison ===")
for i, label in enumerate(labels):
    g, s = gelu_data[i], silu_data[i]
    print(f"{label:8s}  GELU: {g.mean():.1f} +/- {1.96*g.std()/np.sqrt(len(g)):.1f}%  "
          f"SiLU: {s.mean():.1f} +/- {1.96*s.std()/np.sqrt(len(s)):.1f}%  "
          f"seeds: {list(s)}")

# --- Figure ---
fig, ax = plt.subplots(figsize=(6.75, 3.5))

bar_width = 0.30
gap_between_tasks = 0.8  # extra gap between task groups

# Build x positions with gaps between task groups
x_positions = []
pos = 0
for i in range(n_groups):
    x_positions.append(pos)
    if i % 2 == 1 and i < n_groups - 1:  # after each MoE-GLU, add gap
        pos += 1 + gap_between_tasks
    else:
        pos += 1
x = np.array(x_positions)

gelu_color = '#4878CF'
silu_color = '#E8873A'

gelu_means = [d.mean() for d in gelu_data]
silu_means = [d.mean() for d in silu_data]
gelu_ci = [1.96 * d.std() / np.sqrt(len(d)) for d in gelu_data]
silu_ci = [1.96 * d.std() / np.sqrt(len(d)) for d in silu_data]

bars1 = ax.bar(x - bar_width/2, gelu_means, bar_width, yerr=gelu_ci,
               color=gelu_color, edgecolor='white', linewidth=0.5,
               capsize=2.5, error_kw={'linewidth': 0.8, 'capthick': 0.8},
               label='GELU', zorder=2)
bars2 = ax.bar(x + bar_width/2, silu_means, bar_width, yerr=silu_ci,
               color=silu_color, edgecolor='white', linewidth=0.5,
               capsize=2.5, error_kw={'linewidth': 0.8, 'capthick': 0.8},
               label='SiLU', zorder=2)

# Overlay individual seed dots
np.random.seed(0)
for i in range(n_groups):
    jitter_g = np.random.uniform(-0.05, 0.05, size=len(gelu_data[i]))
    ax.scatter(x[i] - bar_width/2 + jitter_g, gelu_data[i],
               color='black', s=10, zorder=3, alpha=0.55, linewidths=0)
    jitter_s = np.random.uniform(-0.05, 0.05, size=len(silu_data[i]))
    ax.scatter(x[i] + bar_width/2 + jitter_s, silu_data[i],
               color='black', s=10, zorder=3, alpha=0.55, linewidths=0)

# Y-axis
ax.set_ylim(0, 109)
ax.set_yticks([0, 20, 40, 60, 80, 100])

# Annotate modadd GLU SiLU failure: arrow from text to the outlier dot
fail_idx = 2  # Mod-Add GLU
# Position annotation in the empty space between Mod-Add MoE-GLU and Histogram GLU
mid_gap = (x[3] + x[4]) / 2  # midpoint of the gap between groups
ax.annotate('1/5 seeds collapses\nto 2.1% accuracy',
            xy=(x[fail_idx] + bar_width/2, 4),
            xytext=(mid_gap - 0.1, 50),
            ha='center', va='bottom', fontsize=6.5,
            color='#C44E52',
            arrowprops=dict(arrowstyle='->', color='#C44E52',
                            lw=0.9, shrinkA=0, shrinkB=3,
                            connectionstyle='arc3,rad=-0.15'))

# X-axis labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)

# Task group brackets/labels at top
task_groups = [('Add-7', 0, 1), ('Mod-Add', 2, 3), ('Histogram', 4, 5)]
for task_name, i_start, i_end in task_groups:
    mid = (x[i_start] + x[i_end]) / 2
    ax.text(mid, 109, task_name, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# Task group separators
for i_start, i_end in [(1, 2), (3, 4)]:
    sep = (x[i_start] + x[i_end]) / 2
    ax.axvline(x=sep, color='#dddddd', linewidth=0.7, linestyle='-', zorder=0)

ax.set_ylabel('Accuracy (%)', fontsize=9.5)

# Reference line at 100%
ax.axhline(y=100, color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)

# Clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
ax.tick_params(width=0.6, labelsize=8)

ax.legend(fontsize=8, loc='lower left', frameon=False, ncol=2)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

outdir = "/Users/gabesmithline/Desktop/moe-expressivity/figures"
fig.savefig(f"{outdir}/fig_silu_comparison.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{outdir}/fig_silu_comparison.pdf", dpi=300, bbox_inches='tight')
print(f"\nSaved to {outdir}/fig_silu_comparison.png and .pdf")
