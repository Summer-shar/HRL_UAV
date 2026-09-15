# Figure Integration Guide for HRL-UAV Paper

## Quick Start

### 1. Generate Sample Data (for testing)
```bash
python scripts/generate_sample_data.py results/sample_figures
```

### 2. Generate Figures
```bash
python scripts/generate_paper_figures.py results/sample_figures
```

### 3. View Generated Figures
Figures will be saved to:
```
paper_current_20260907/figures/
├── fig4x_main_comparison.pdf
├── fig4x_main_comparison.png
├── fig4x_main_comparison.svg
├── fig4y_j_decomposition.pdf
├── fig4y_j_decomposition.png
├── fig4y_j_decomposition.svg
├── fig4z_trajectory.pdf
├── fig4z_trajectory.png
├── fig4z_trajectory.svg
├── fig5x_training_curves.pdf
├── fig5x_training_curves.png
├── fig5x_training_curves.svg
├── fig5y_ablation.pdf
├── fig5y_ablation.png
├── fig5y_ablation.svg
└── figure_manifest.json
```

## Using Real Mission Data

The figure generation script reads from mission result directories. Point it to your completed mission results:

```bash
python scripts/generate_paper_figures.py results/mission_paper_comparison
```

Required input files:
- `summary.csv` - aggregated metrics by method
- `per_seed.csv` - per-seed results
- `learning_curves.json` - training curve data (optional)
- `ablation.json` - ablation study data (optional)

## Manual Figure Updates

If you need to modify individual figures, use the Python API directly:

```python
from scripts.generate_paper_figures import (
    setup_matplotlib,
    plot_fig4x_main_comparison,
    save_figure
)

setup_matplotlib()
fig = plot_fig4x_main_comparison(summary_data)
save_figure(fig, 'fig4x_main_comparison')
```

## LaTeX Compilation

After generating figures, compile the paper:

```bash
cd paper_current_20260907
latexmk -pdf main.tex
```

Or with xelatex:
```bash
cd paper_current_20260907
xelatex -interaction=nonstopmode main.tex
```

## Color Palette Reference

| Method Type | Color | Hex Code |
|------------|-------|----------|
| Learning (SAC) | Blue | #2868a0 |
| Learning (PPO) | Light Blue | #4a90d9 |
| Learning (Joint) | Pale Blue | #7ab8f5 |
| Baseline (CVX) | Dark Gray | #555555 |
| Baseline (Fixed) | Gray | #777777 |
| Baseline (Random) | Light Gray | #999999 |
| Oracle (Predictive) | Green | #2d8a4e |
| Significant | Red | #b32430 |

## Troubleshooting

### Fonts not embedded
Ensure `pdf.fonttype: 42` is set in matplotlib configuration.

### Chinese characters not rendering
Use `DejaVu Sans` or `SimHei` as fallback fonts:
```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
```

### Figure layout issues
Check `savefig.bbox='tight'` and `savefig.pad_inches=0.1` settings.

### DPI too low
Default is 300 DPI. Adjust in `save_figure()` function if needed.
