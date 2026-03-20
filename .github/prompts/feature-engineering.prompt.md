You are extending Phi-nance automated feature engineering.

Goals:
1. Improve predictive quality of discovered features without increasing API usage.
2. Keep outputs reproducible (fixed seeds, explicit configs).
3. Preserve backward compatibility with agents that disable discovered features.

When proposing code:
- Include autoencoder + GP changes separately in commits/PR sections.
- Add tests for shape, stability, and registry persistence.
- Document runtime implications and fallback behavior.
