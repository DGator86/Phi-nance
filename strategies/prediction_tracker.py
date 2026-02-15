"""
Prediction Tracker
------------------
Mixin for Lumibot strategies that records daily directional predictions
and scores them after the backtest completes.

Each strategy calls `self.record_prediction(symbol, signal, price)` once
per trading iteration.  After the backtest, call
`compute_prediction_accuracy(strategy_instance)` to get a full scorecard.

Signals:
    "UP"      — strategy expects price to rise by next check-in
    "DOWN"    — strategy expects price to fall by next check-in
    "NEUTRAL" — no directional view (not scored)
"""


class PredictionMixin:
    """Mix into any Lumibot Strategy subclass to gain prediction tracking."""

    def _init_predictions(self):
        if not hasattr(self, "_prediction_log"):
            self._prediction_log = []

    def record_prediction(self, symbol, signal, price):
        """Record a directional prediction for the current iteration.

        Args:
            symbol: The ticker being predicted.
            signal: "UP", "DOWN", or "NEUTRAL".
            price:  The price at the time the prediction is made.
        """
        self._init_predictions()
        self._prediction_log.append({
            "date": self.get_datetime(),
            "symbol": symbol,
            "signal": signal,
            "price": price,
        })


def compute_prediction_accuracy(strategy):
    """Post-backtest: compare each prediction to what actually happened.

    For each prediction at time T with price P, we look at the *next*
    prediction's price P' for the same symbol.  If signal was "UP" and
    P' > P, that's a hit.  If "DOWN" and P' < P, also a hit.  "NEUTRAL"
    predictions are skipped.

    Returns a dict with accuracy metrics and the full scored log.
    """
    log = getattr(strategy, "_prediction_log", [])
    if not log:
        return _empty_scorecard()

    # Group by symbol so we compare consecutive prices per-symbol
    by_symbol = {}
    for entry in log:
        by_symbol.setdefault(entry["symbol"], []).append(entry)

    scored = []
    for symbol, entries in by_symbol.items():
        entries.sort(key=lambda e: e["date"])
        for i in range(len(entries) - 1):
            curr = entries[i]
            nxt = entries[i + 1]
            if curr["signal"] == "NEUTRAL":
                continue

            actual_move = nxt["price"] - curr["price"]
            if actual_move == 0:
                correct = False  # flat = not a correct directional call
            elif curr["signal"] == "UP":
                correct = actual_move > 0
            else:  # DOWN
                correct = actual_move < 0

            scored.append({
                "date": curr["date"],
                "symbol": curr["symbol"],
                "signal": curr["signal"],
                "price": curr["price"],
                "next_price": nxt["price"],
                "actual_move": actual_move,
                "correct": correct,
            })

    if not scored:
        return _empty_scorecard()

    total = len(scored)
    hits = sum(1 for s in scored if s["correct"])
    misses = total - hits

    up_preds = [s for s in scored if s["signal"] == "UP"]
    down_preds = [s for s in scored if s["signal"] == "DOWN"]

    up_hits = sum(1 for s in up_preds if s["correct"])
    down_hits = sum(1 for s in down_preds if s["correct"])

    # Avg magnitude of move when correct vs incorrect
    correct_moves = [abs(s["actual_move"]) for s in scored if s["correct"]]
    incorrect_moves = [abs(s["actual_move"]) for s in scored if not s["correct"]]
    avg_correct_magnitude = (
        sum(correct_moves) / len(correct_moves) if correct_moves else 0
    )
    avg_incorrect_magnitude = (
        sum(incorrect_moves) / len(incorrect_moves) if incorrect_moves else 0
    )

    return {
        "total_predictions": total,
        "hits": hits,
        "misses": misses,
        "accuracy": hits / total if total else 0,
        "up_predictions": len(up_preds),
        "up_accuracy": up_hits / len(up_preds) if up_preds else 0,
        "down_predictions": len(down_preds),
        "down_accuracy": down_hits / len(down_preds) if down_preds else 0,
        "avg_correct_magnitude": avg_correct_magnitude,
        "avg_incorrect_magnitude": avg_incorrect_magnitude,
        "edge": avg_correct_magnitude - avg_incorrect_magnitude,
        "scored_log": scored,
    }


def _empty_scorecard():
    return {
        "total_predictions": 0,
        "hits": 0,
        "misses": 0,
        "accuracy": 0,
        "up_predictions": 0,
        "up_accuracy": 0,
        "down_predictions": 0,
        "down_accuracy": 0,
        "avg_correct_magnitude": 0,
        "avg_incorrect_magnitude": 0,
        "edge": 0,
        "scored_log": [],
    }
