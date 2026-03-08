# External Options & Market-Data Landscape

This guide tracks external projects relevant to Phi-nance options/data evolution and documents practical integration priorities.

## Candidate Repositories

| Repository | Primary value for Phi-nance | Practical use | Integration risk |
|---|---|---|---|
| `MarketDataApp/sdk-py` | Clean Python SDK for US equities/options snapshots | Optional chain/quote enrichment behind env-gated connector | Vendor schema/version coupling |
| `OpenBB-finance/OpenBB` | Wide multi-provider data abstraction | Meta-provider exploration for breadth-heavy workflows | Larger dependency footprint |
| `yugedata/Options_Data_Science` | Options analytics workflow patterns | Feature engineering and exploratory analysis references | Notebook-heavy, production hardening needed |
| `SamPom100/OptionsAnalyzer` | Visual options analytics ideas | Heatmaps/surface visual patterns for Streamlit pages | UI conventions require adaptation |
| `binance/binance-public-data` | Large free crypto datasets | Alternative regime experiments and stress testing | Data volume/storage costs |
| `nuglifeleoji/Options-Analytics-Agent` | Agentic options-analysis architecture | Future PhiAI agent orchestration reference | LLM orchestration complexity |

## Suggested Adoption Order

1. **Low-friction data wins first**: optional MarketDataApp connector hardening.
2. **Visualization and analytics enhancement**: port selected options analytics/UI concepts.
3. **Provider breadth**: evaluate OpenBB when multi-provider abstraction becomes mandatory.
4. **Agentic layer**: adopt advanced orchestration patterns after options-core maturity.

## Integration Principles

- Keep third-party connectors optional and resilient to missing keys.
- Normalize external schemas before entering backtest/analytics pipelines.
- Preserve offline/local workflows with sample or fallback data paths.
- Add tests around provider boundaries to guard against upstream API drift.
