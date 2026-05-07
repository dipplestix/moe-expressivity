"""Strip in-figure arrow/text annotations from existing PNGs by overwriting
the affected regions with the panel background color (white).

Targets:
  - paper/figures/fig3_num_experts.png  ("E=1 has aux loss but doesn't help" + arrow)
  - paper/figures/fig7_width_scaling.png ("1.1x", "2.2x", "2.7x" labels)
  - paper/figures/fig_silu_comparison.png ("1/5 seeds collapses to 2.1% accuracy" + arrow)

The bounding boxes below were manually identified from the rendered figures.
We overwrite with white (255, 255, 255). Each box is (x0, y0, x1, y1) in image-pixel
coordinates (origin top-left).
"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("<PATH_TO_REPO>")
WHITE = (255, 255, 255)

JOBS = [
    {
        "path": ROOT / "paper" / "figures" / "fig3_num_experts.png",
        # Image is 3551x1536. Panel (a) "Grokking Speed" is the left ~half.
        # Annotation text + arrow in upper-left of (a). Conservative bbox.
        "boxes": [(110, 100, 1050, 460)],
    },
    {
        "path": ROOT / "paper" / "figures" / "fig7_width_scaling.png",
        # Image is 2052x1452. Three "Nx" labels above bars at d=64, d=128, d=256.
        # The bars are at roughly x positions: bar centers 460, 1080, 1700.
        # Each label sits above its bar pair. Wipe a strip across the top.
        # Label "1.1x" near (440, 200), "2.2x" near (1000, 90), "2.7x" near (1620, 145).
        # Conservative single horizontal strip at the top of the plot area:
        "boxes": [
            (300, 60, 600, 300),    # 1.1x area
            (900, 60, 1180, 200),   # 2.2x area
            (1500, 60, 1780, 240),  # 2.7x area
        ],
    },
    {
        "path": ROOT / "paper" / "figures" / "fig_silu_comparison.png",
        # Image is 1994x974. "1/5 seeds collapses to 2.1% accuracy" + curved red arrow.
        # Located between Mod-Add and Histogram panels, middle-right of (b).
        "boxes": [
            (920, 380, 1280, 760),  # text + arrow region
        ],
    },
]


def main():
    for job in JOBS:
        path = job["path"]
        if not path.exists():
            print(f"  skip {path.name} (not found)")
            continue
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in job["boxes"]:
            draw.rectangle(box, fill=WHITE)
        img.save(path)
        print(f"  cleaned {path.relative_to(ROOT)} ({len(job['boxes'])} regions wiped)")


if __name__ == "__main__":
    main()
