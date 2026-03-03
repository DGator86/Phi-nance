# Phi-nance Visual Spec — Dark Purple/Orange Theme

## Color Palette (Hex)

| Role | Hex | Usage |
|------|-----|-------|
| **Primary (Purple)** | `#a855f7` | CTAs, active states, key metrics |
| **Primary Dark** | `#7c3aed` | Gradients, hover |
| **Primary Light** | `#c084fc` | Highlights |
| **Accent (Orange)** | `#f97316` | Warnings, secondary CTAs, delta positive |
| **Background** | `#0f0f12` | Main app background |
| **Card/Surface** | `#1a1a1f` | Cards, expanders, secondary surfaces |
| **Border** | `rgba(168,85,247,0.15)` | Subtle borders |
| **Text Primary** | `#e4e4e7` | Body text |
| **Success** | `#22c55e` | Profit, success states |
| **Error** | `#ef4444` | Loss, errors |
| **Neutral** | `#94a3b8` | Muted text |

## Chart Colors (Plotly/Altair)

Use for multi-line charts (equity curve, indicator comparison):

```python
CHART_COLORS = [
    "#a855f7",  # Primary purple
    "#f97316",  # Orange
    "#22c55e",  # Green
    "#06b6d4",  # Cyan
    "#eab308",  # Yellow
]
```

## Streamlit Config

Already in `.streamlit/config.toml`:

- `primaryColor = "#a855f7"`
- `backgroundColor = "#0f0f12"`
- `secondaryBackgroundColor = "#1a1a1f"`
- `textColor = "#e4e4e7"`
