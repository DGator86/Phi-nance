"""
phinance.phibot — Post-backtest AI review engine.

Usage
-----
    from phinance.phibot.reviewer import review_backtest

    review = review_backtest(
        ohlcv          = df,
        results        = result.to_dict(),
        prediction_log = result.prediction_log,
        indicators     = indicators,
        blend_weights  = weights,
        blend_method   = "weighted_sum",
        config         = {"symbols": ["SPY"]},
    )
    print(review.summary)
"""

from phinance.phibot.reviewer import review_backtest, BacktestReview

__all__ = ["review_backtest", "BacktestReview"]
