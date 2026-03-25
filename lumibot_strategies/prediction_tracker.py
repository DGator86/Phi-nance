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


# ──────────────────────────────────────────────────────────────────────────────
# Options prediction tracking
# ──────────────────────────────────────────────────────────────────────────────

def compute_options_scorecard(strategy):
    """
    Score options trade predictions recorded via record_options_prediction().

    For each recorded options trade (signal == 'OPTIONS'), we score it by
    comparing the entry spot price to the spot price at the *next* OPTIONS
    prediction for the same symbol.  We use structure-level expected direction:
      - Bullish structures (long_call, bull_*): correct if price rises
      - Bearish structures (long_put, bear_*):  correct if price falls
      - Non-directional (straddle, condor, etc.): correct if |move| > avg spread

    Returns a dict with overall and per-structure accuracy, avg net_credit, P&L.
    """
    log = [e for e in getattr(strategy, "_prediction_log", []) if e.get("signal") == "OPTIONS"]
    if not log:
        return _empty_options_scorecard()

    by_symbol = {}
    for entry in log:
        by_symbol.setdefault(entry["symbol"], []).append(entry)

    _BULLISH = {"long_call", "bull_call_spread", "bull_put_spread", "collar", "covered_call"}
    _BEARISH = {"long_put", "bear_put_spread", "bear_call_spread"}

    scored = []
    by_structure: dict = {}

    for symbol, entries in by_symbol.items():
        entries.sort(key=lambda e: e["date"])
        for i in range(len(entries) - 1):
            curr = entries[i]
            nxt  = entries[i + 1]
            move = nxt["price"] - curr["price"]
            structure = curr.get("structure", "unknown")
            max_loss  = curr.get("max_loss", 0) or 0
            net_credit = curr.get("net_credit", 0) or 0
            beqs = curr.get("breakeven", []) or []

            if structure in _BULLISH:
                correct = move > 0
                simple_pnl = move * 100 * 0.5  # rough delta = 0.5 × 100 shares
            elif structure in _BEARISH:
                correct = move < 0
                simple_pnl = -move * 100 * 0.5
            else:
                # Non-directional: win if price moves beyond a breakeven or stays in range
                avg_be = sum(beqs) / len(beqs) if beqs else curr["price"]
                if net_credit > 0:
                    # Short premium: win if price stays near avg_be
                    correct = abs(curr["price"] - avg_be) < abs(move) * 0.5
                else:
                    # Long vol: win if price moves a lot
                    correct = abs(move) > abs(avg_be - curr["price"]) * 0.3
                simple_pnl = net_credit if correct else -abs(max_loss)

            record = {
                "date":        curr["date"],
                "symbol":      symbol,
                "structure":   structure,
                "level":       curr.get("level", "?"),
                "regime":      curr.get("regime", "?"),
                "vol_regime":  curr.get("vol_regime", "?"),
                "gex_regime":  curr.get("gex_regime", "?"),
                "confidence":  curr.get("confidence", 0.0),
                "price":       curr["price"],
                "next_price":  nxt["price"],
                "move":        move,
                "correct":     correct,
                "est_pnl":     simple_pnl,
                "net_credit":  net_credit,
            }
            scored.append(record)

            if structure not in by_structure:
                by_structure[structure] = {"n": 0, "wins": 0, "total_pnl": 0.0}
            by_structure[structure]["n"]         += 1
            by_structure[structure]["wins"]      += int(correct)
            by_structure[structure]["total_pnl"] += simple_pnl

    if not scored:
        return _empty_options_scorecard()

    total = len(scored)
    hits  = sum(1 for s in scored if s["correct"])

    structure_summary = {
        k: {
            "n":        v["n"],
            "accuracy": v["wins"] / max(v["n"], 1),
            "avg_pnl":  v["total_pnl"] / max(v["n"], 1),
        }
        for k, v in by_structure.items()
    }

    return {
        "total_options_predictions": total,
        "hits":                      hits,
        "misses":                    total - hits,
        "accuracy":                  hits / total,
        "avg_est_pnl":               sum(s["est_pnl"] for s in scored) / total,
        "by_structure":              structure_summary,
        "scored_log":                scored,
    }


def _empty_options_scorecard():
    return {
        "total_options_predictions": 0,
        "hits":    0,
        "misses":  0,
        "accuracy": 0,
        "avg_est_pnl": 0,
        "by_structure": {},
        "scored_log": [],
    }
