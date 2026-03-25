# QuantConnect + Phi-nance ML inference

## Reality check

QuantConnect **Lean** algorithms run in QC’s environment. A **joblib pickle** trained in Phi-nance (sklearn / hmmlearn / custom `RegimeDetector`) is **not guaranteed** to load or behave identically on QC (versions, native deps, security policy).

Practical patterns:

1. **Export regime labels as CSV** (recommended for v1)  
   Train in Phi-nance, then:

   ```bash
   python scripts/export_regime_predictions_for_qc.py \
     --model path/to/model.pkl --symbol SPY --start ... --end ... \
     --out regimes.csv
   ```

   Upload `regimes.csv` as **custom data** next to `ohlcv.csv` and align on `time` in `QCAlgorithm`.

2. **Provenance bundle**  
   `phi.regime.artifacts.export_regime_bundle_for_quantconnect` copies `.pkl`, metadata JSON, and optional `regimes.csv` into one folder for archiving or research notebooks—not a promise that QC will execute the pickle.

3. **Future**  
   ONNX, pure-numpy parameters, or a **QC-native** re-fit of a simple model on historical `History()` bars.

## Links

- OHLCV export: [quantconnect_deployment.md](quantconnect_deployment.md)
- ML tooling index: [ml_components.md](ml_components.md)
