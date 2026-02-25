# External Options & Market-Data Landscape

This note summarizes external repositories that can complement Phi-nance's options and market-data roadmap.

## Candidate repositories

| Repository | Primary value for Phi-nance | How to use it here | Integration risk |
|---|---|---|---|
| `MarketDataApp/sdk-py` | Clean Python SDK for U.S. equities/options data (real-time + historical) | Add as an optional data connector in `phi.data` and use for options chain ingestion | API key dependency + vendor-specific schemas |
| `yugedata/Options_Data_Science` | End-to-end options analytics patterns (collection, analysis, viz, paper trading) | Borrow feature engineering and analytics workflow ideas for `phi/options` | Notebook-heavy code may need production hardening |
| `binance/binance-public-data` | Large, free crypto historical market data source | Extend data builder to support crypto symbols and regime modeling experiments | Bulk data management/storage concerns |
| `mcdallas/wallstreet` | Lightweight real-time stock/options access utilities | Fast prototyping for options quotes/chains in experiments | Unofficial wrappers can change unexpectedly |
| `OpenBB-finance/OpenBB` | Broad multi-provider financial data platform + tools for analysts/agents | Use as a meta data-provider layer when breadth matters more than minimal dependencies | Heavy dependency surface; versioning discipline needed |
| `nuglifeleoji/Options-Analytics-Agent` | Agentic options analysis architecture (LangGraph, caching, memory) | Reference architecture for a future PhiAI options-agent workflow | Adds LLM orchestration complexity |
| `SamPom100/OptionsAnalyzer` | Strong visual analytics patterns for options surfaces/heatmaps | Reuse visualization ideas in Streamlit (volatility surface, OI heatmaps) | Project-specific plotting conventions may need adaptation |

## Suggested adoption order

1. **Data breadth first:** trial `MarketDataApp/sdk-py` and `binance/binance-public-data` behind optional connectors.
2. **Analytics second:** port selected feature/viz concepts from `Options_Data_Science` and `OptionsAnalyzer`.
3. **Platform scale-up:** evaluate `OpenBB` if a unified provider layer is needed.
4. **Agent layer last:** adapt ideas from `Options-Analytics-Agent` after options backtest primitives are stable.

## Minimal implementation plan for Phi-nance

- Add `phi/data/providers/` abstraction with a common fetch interface (OHLCV, options chain, greeks snapshot).
- Implement one low-risk connector first (`MarketDataApp`), guarded by env flag and graceful fallback.
- Extend `phi/options/` from stub to include normalized options chain schema and caching.
- Add Streamlit panel cards for options OI/IV visualizations inspired by external projects.
- Keep all third-party integrations optional to preserve current lightweight local workflows.
