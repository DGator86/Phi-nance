"""
frontend.streamlit — Phi-nance multi-page Streamlit application.

Pages map directly to the quant workflow:
  1_Dataset_Builder.py    → phinance.data
  2_Indicator_Selection.py → phinance.strategies
  3_Blending.py           → phinance.blending
  4_PhiAI_Optimization.py → phinance.optimization
  5_Backtest_Controls.py  → phinance.backtest
  6_Results.py            → phinance.storage + phinance.backtest.metrics
"""
