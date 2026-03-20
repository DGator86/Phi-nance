# Blending

`phi.blending.blender.blend_signals` combines indicator signals into a composite stream.

## Supported methods

- `weighted_sum`
- `voting`
- `regime_weighted`

## Usage

```python
combined = blend_signals(
    signals,
    method="weighted_sum",
    weights={"RSI": 0.6, "MACD": 0.4},
)
```

## Extending with a new blend method

1. Add implementation in `phi/blending/blender.py` (or delegated module).
2. Add method validation to accepted method set.
3. Update UI option lists if needed.
4. Add tests for correctness and edge cases.
