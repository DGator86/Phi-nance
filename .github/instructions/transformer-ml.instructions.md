---
name: transformer-ml
description: Implementation and validation guidance for transformer-based market feature extraction.
maturity: beta
schema: hve-core/instruction/v1
---

# Transformer ML Instructions

1. Prefer deterministic data splits for reproducible training runs.
2. Persist feature normalization stats with each model checkpoint.
3. Keep default model footprint CPU-friendly for agent inference.
4. Add or update tests whenever feature columns or architecture contracts change.
5. Ensure agent integration is optional via config flags and graceful fallback behavior.
