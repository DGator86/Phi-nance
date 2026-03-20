---
name: strategy-proposal
description: Propose candidate indicator combinations and explain expected behavior by market regime.
inputs:
  - name: market_context
    description: Summary of instruments, timeframe, market microstructure notes, and current regime priors.
    required: true
  - name: constraints
    description: Risk, turnover, latency, and capital constraints for strategy deployment.
    required: true
maturity: experimental
schema: hve-core/prompt/v1
---

# Strategy Proposal

You are designing a new Phi-nance strategy variant.

## Inputs

### Market Context
${input:market_context}

### Constraints
${input:constraints}

## Output Requirements

Provide:

1. **Hypothesis**
   - Core edge and why it may persist.
2. **Indicator Stack (3 candidates)**
   - For each candidate, list indicator set, parameter ranges, and signal-combination logic.
3. **Regime Expectations**
   - Expected performance in trending, mean-reverting, and volatile/choppy regimes.
4. **Risk Controls**
   - Position sizing, stop logic, and exposure caps.
5. **Backtest Design**
   - Dataset period, walk-forward split, metrics, and failure criteria.
6. **Implementation Notes**
   - Required data fields and complexity estimate.

Constraints:
- Keep all proposals within the provided constraints.
- Mark assumptions clearly.
- Prioritize interpretable features over opaque complexity.
