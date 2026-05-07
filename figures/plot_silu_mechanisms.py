"""
Figure: SiLU activation robustness check for three main mechanistic findings.
3-panel figure for NeurIPS full-width (6.75 inches).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Colors
GELU = '#4878CF'
SILU = '#E8873A'

fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.0))
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

# ── Panel (a): H1 Redistribution ──
ax = axes[0]
# no-FFN accuracy (the key metric showing redistribution)
labels = ['GLU', 'MoE-GLU']
gelu_means = [12.0, 30.7]
gelu_errs  = [3.4, 5.6]
silu_means = [13.7, 32.5]
silu_errs  = [4.0, 6.0]

x = np.arange(len(labels))
w = 0.32
bars1 = ax.bar(x - w/2, gelu_means, w, yerr=gelu_errs, color=GELU, edgecolor='white',
               linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8}, label='GELU', zorder=3)
bars2 = ax.bar(x + w/2, silu_means, w, yerr=silu_errs, color=SILU, edgecolor='white',
               linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8}, label='SiLU', zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('No-FFN accuracy (%)')
ax.set_title('(a) Redistribution (H1)', fontsize=9, fontweight='semibold')
ax.set_ylim(0, 50)
ax.legend(fontsize=6.5, frameon=False, loc='upper left')

# ── Panel (b): H2 Fourier Opacity ──
ax = axes[1]

# Data: per-neuron and top-PC for each variant x activation
categories = ['per-\nneuron', 'top-\nPC', 'per-\nneuron', 'top-\nPC']
gelu_m = [0.048, 0.495, 0.123, 0.495]
gelu_e = [0.036, 0.006, 0.069, 0.005]
silu_m = [0.171, 0.481, 0.119, 0.455]
silu_e = [0.151, 0.026, 0.064, 0.070]

x = np.arange(len(categories))
w = 0.32
ax.bar(x - w/2, gelu_m, w, yerr=gelu_e, color=GELU, edgecolor='white',
       linewidth=0.5, capsize=2.5, error_kw={'linewidth': 0.8}, label='GELU', zorder=3)
ax.bar(x + w/2, silu_m, w, yerr=silu_e, color=SILU, edgecolor='white',
       linewidth=0.5, capsize=2.5, error_kw={'linewidth': 0.8}, label='SiLU', zorder=3)

# Divider between GLU and MoE groups
ax.axvline(1.5, color='gray', lw=0.5, ls=':', zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=6)
ax.set_ylabel('Fourier concentration')
ax.set_title('(b) Fourier opacity (H2)', fontsize=9, fontweight='semibold')
ax.set_ylim(0, 0.7)
# Group labels
ax.text(0.5, -0.28, 'GLU', ha='center', va='top', fontsize=7, fontweight='semibold', transform=ax.get_xaxis_transform())
ax.text(2.5, -0.28, 'MoE-GLU', ha='center', va='top', fontsize=7, fontweight='semibold', transform=ax.get_xaxis_transform())
ax.legend(fontsize=6.5, frameon=False, loc='upper left')

# ── Panel (c): H3 Routing MI ──
ax = axes[2]

gelu_seeds = np.array([0.4824, 0.2575, 0.2233, 0.1505, 0.1499])
silu_seeds = np.array([0.4668, 0.2646, 0.2331, 0.1532, 0.1605])
gelu_mean = gelu_seeds.mean()
silu_mean = silu_seeds.mean()

# Paired dot plot
for i in range(len(gelu_seeds)):
    ax.plot([0, 1], [gelu_seeds[i], silu_seeds[i]], color='gray', lw=0.7, zorder=2)
ax.scatter(np.zeros(len(gelu_seeds)), gelu_seeds, color=GELU, s=28, zorder=4, edgecolors='white', linewidths=0.4)
ax.scatter(np.ones(len(silu_seeds)), silu_seeds, color=SILU, s=28, zorder=4, edgecolors='white', linewidths=0.4)

# Mean bars
bar_hw = 0.18
ax.plot([-bar_hw, bar_hw], [gelu_mean, gelu_mean], color=GELU, lw=2.2, zorder=5)
ax.plot([1 - bar_hw, 1 + bar_hw], [silu_mean, silu_mean], color=SILU, lw=2.2, zorder=5)

# Mean text
ax.text(0, gelu_mean + 0.025, f'{gelu_mean:.3f}', ha='center', va='bottom', fontsize=6.5, color=GELU)
ax.text(1, silu_mean + 0.025, f'{silu_mean:.3f}', ha='center', va='bottom', fontsize=6.5, color=SILU)

ax.set_xticks([0, 1])
ax.set_xticklabels(['GELU', 'SiLU'])
ax.set_ylabel('Routing NMI')
ax.set_title('(c) Routing MI (H3)', fontsize=9, fontweight='semibold')
ax.set_ylim(0, 0.58)
ax.set_xlim(-0.5, 1.5)

# ── Global styling ──
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

plt.tight_layout(w_pad=1.8)
fig.savefig('<PATH_TO_REPO>/figures/fig_silu_mechanisms.png', dpi=300, bbox_inches='tight')
fig.savefig('<PATH_TO_REPO>/figures/fig_silu_mechanisms.pdf', bbox_inches='tight')
print('Saved PNG and PDF.')
