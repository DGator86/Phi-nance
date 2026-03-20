Implement distributed backtesting in Phi-nance using Ray.

Goals:
1. Keep non-distributed mode fully functional.
2. Add a `DistributedBacktestRunner` for parallel sweeps.
3. Enable GP population fitness evaluation to run in parallel batches.
4. Add tests that compare distributed outputs to sequential outputs.
5. Update documentation and config defaults.
