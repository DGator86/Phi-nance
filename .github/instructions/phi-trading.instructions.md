---
name: phi-trading
description: Phi-nance standards for quantitative trading code, agent collaboration, and HVE artifact quality.
applyTo:
  - "**/*.py"
  - "**/*.ipynb"
  - ".github/agents/**/*.md"
  - ".github/prompts/**/*.md"
  - ".github/instructions/**/*.md"
owner: phi-nance
maturity: experimental
schema: hve-core/instruction/v1
---

# Phi-trading Instructions

## General Principles

- Use explicit type hints on all public functions, methods, and class attributes.
- Add concise docstrings for all public modules, classes, and functions.
- Keep functions focused and composable; avoid hidden side effects.
- Prefer deterministic computations for backtest/research paths.
- For non-trivial changes, include a runnable validation command in the PR notes.

## Financial Logic

- Keep financial units explicit (price, return, volatility, notional, basis points).
- Validate indicator inputs (window length, missing OHLCV fields, NaN handling) before computation.
- Greeks and risk metrics must declare assumptions (e.g., annualization basis, model family, sampling frequency).
- Backtest metrics must include at minimum: cumulative return, Sharpe, max drawdown, win rate, and turnover.
- Never blend live and backtest data paths without a clear feature flag or environment guard.

## Agent Behavior

- For complex work, follow RPI sequencing: **Research -> Plan -> Implement**.
- Use `runSubagent` for delegated deep work, especially:
  - research synthesis,
  - task decomposition,
  - memory/context summarization.
- Preserve traceability by citing generated artifacts (plans, reports, decisions) in downstream outputs.
- Escalate ambiguity explicitly and propose bounded options with trade-offs.

## Validation

- Every agent, instruction, and prompt file must include valid YAML frontmatter.
- Frontmatter should include `name`, `description`, `maturity`, and schema metadata.
- Artifact maturity progression:
  - `experimental` -> `beta` -> `stable`
- CI must run HVE validation over:
  - `.github/agents`
  - `.github/instructions`
  - `.github/prompts`
