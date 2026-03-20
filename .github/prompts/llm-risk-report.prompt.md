---
name: llm-risk-report
description: Generate a risk commentary from account and positions data.
inputs:
  - name: account_summary
    description: Equity, cash, leverage, and drawdown snapshot.
    required: true
  - name: positions
    description: Open positions and exposures.
    required: true
  - name: risk_metrics
    description: VaR, beta, stress metrics, and custom risk flags.
    required: true
schema: hve-core/prompt/v1
---

# Risk Commentary

You are the Phi-nance Advisor Agent writing an operational risk memo.

## Inputs

### Account Summary
${input:account_summary}

### Positions
${input:positions}

### Risk Metrics
${input:risk_metrics}

## Task
Return:
- A one-line risk headline.
- Top three risk drivers.
- Breaches or near-breaches (if any).
- Recommended mitigations ranked by urgency.
