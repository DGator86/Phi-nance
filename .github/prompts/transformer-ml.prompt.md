---
name: transformer-ml-prompt
description: Prompt scaffold for extending transformer-based predictive features in Phi-nance.
maturity: beta
schema: hve-core/prompt/v1
---

You are extending the Phi-nance transformer ML stack.

Goals:
1. Improve sequence feature quality while preserving low-latency inference.
2. Keep compatibility with existing agent state builders.
3. Propose measurable experiments with explicit validation metrics.

Checklist:
- Describe data features and any new vendors/sources.
- Specify model shape and expected inference latency impact.
- Provide migration notes for checkpoint/schema changes.
- Include concrete test updates (unit + integration).
- Document failure modes and fallback behavior.
