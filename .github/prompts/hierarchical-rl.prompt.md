# Prompt: Design a hierarchical RL option

You are designing a new hierarchical RL option for Phi-nance.

Provide:
- Option name and responsibility.
- Initiation condition using available context fields.
- Termination condition and max duration.
- Required low-level policy interface (`act(state, deterministic)`).
- Reward shaping impacts and possible penalties.
- Unit test cases for initiation, termination, and interruption.

Target files:
- `phinance/rl/hierarchical/options.py`
- `phinance/rl/hierarchical/wrappers.py`
- `tests/unit/rl/hierarchical/`
