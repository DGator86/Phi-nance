"""Cox-Ross-Rubinstein binomial tree pricing utilities."""

from __future__ import annotations

import math


def price_american(option_type: str, S: float, K: float, T: float, r: float, sigma: float, n: int = 100) -> float:
    """Price an American option with a CRR binomial tree."""
    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if n <= 0:
        raise ValueError("n must be positive")

    dt = T / n
    u = math.exp(sigma * math.sqrt(dt)) if sigma > 0 else 1.0
    d = 1.0 / u if u != 0 else 0.0
    disc = math.exp(-r * dt)

    if abs(u - d) < 1e-12:
        p = 0.5
    else:
        p = (math.exp(r * dt) - d) / (u - d)
        p = min(max(p, 0.0), 1.0)

    values = [0.0] * (n + 1)
    for j in range(n + 1):
        stock = S * (u ** j) * (d ** (n - j))
        intrinsic = max(stock - K, 0.0) if option_type == "call" else max(K - stock, 0.0)
        values[j] = intrinsic

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            stock = S * (u ** j) * (d ** (i - j))
            hold = disc * (p * values[j + 1] + (1 - p) * values[j])
            exercise = max(stock - K, 0.0) if option_type == "call" else max(K - stock, 0.0)
            values[j] = max(hold, exercise)

    return values[0]


def price_european(option_type: str, S: float, K: float, T: float, r: float, sigma: float, n: int = 100) -> float:
    """Price a European option with a CRR tree (no early exercise)."""
    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    dt = T / n
    u = math.exp(sigma * math.sqrt(dt)) if sigma > 0 else 1.0
    d = 1.0 / u if u != 0 else 0.0
    disc = math.exp(-r * dt)

    if abs(u - d) < 1e-12:
        p = 0.5
    else:
        p = (math.exp(r * dt) - d) / (u - d)
        p = min(max(p, 0.0), 1.0)

    values = [0.0] * (n + 1)
    for j in range(n + 1):
        stock = S * (u ** j) * (d ** (n - j))
        values[j] = max(stock - K, 0.0) if option_type == "call" else max(K - stock, 0.0)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            values[j] = disc * (p * values[j + 1] + (1 - p) * values[j])

    return values[0]
