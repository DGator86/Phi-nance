# LLM Integration Instructions

- Keep LLM usage optional and failure-tolerant.
- Do not block trading loops if model calls fail.
- Use prompt templates from `.github/prompts/*.prompt.md`.
- Compress oversized prompt context before sending to providers.
- Prefer deterministic fallbacks for local/offline testing.
