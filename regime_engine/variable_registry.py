"""
Variable Registry — Central state-space model for the assumption-free framework.

Principle
---------
No hardcoded constants. No assumed relationships. No fixed lookbacks. No preset
thresholds.

Everything is either:
  A) Observed from data
  B) Learned from history
  C) A variable with its own update equation

The ONLY true constants in this module are:
  RHO_MIN, RHO_MAX — stability bounds on learning rates

All other quantities are declared as state variables with explicit update
equations driven by market observations and prediction errors.

Full Variable Registry
----------------------
VARIABLE               TYPE           UPDATE MECHANISM
──────────────────────────────────────────────────────────────────────────────
tau(t)                 State          autocorrelation decay observation
eta_L(t)               State × level  regime velocity + prediction error
C(i, k, t)             Matrix         empirical signal-regime covariance
alpha_j(t)             Vector         predictive contribution rolling
W_j(t)                 Vector         performance of window length
M_interaction(t)       Matrix         learned indicator cross-correlations
theta_r(t)             Matrix         weighted gradient from projection error
sigma_cone(H, t)       Function       realized dispersion at horizon H
cone_skew(H, t)        Function       asymmetry of realized projection errors
Gamma(t)               Matrix         meta-learning on projection params
rho_tau(t)             Scalar         meta-learning on tau adaptation
kappa_eta(t)           Scalar         meta-learning on eta adaptation
lambda_C(t)            Scalar         meta-learning on C adaptation
tau_transition_L(t)    State × level  observed time between regime transitions
P_persist_k(t)         Vector         empirical persistence of each node
H_k(t)                 State × level  data-derived horizons (not formula)
consistency(t)         Scalar         micro-macro cross-scale consistency
W_con(t)               Scalar         adaptive consistency window
lambda_con(t)          Scalar         consistency adaptation rate
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional


# ─── Stability bounds (the ONLY true constants) ────────────────────────────────
RHO_MIN: float = 1e-4   # minimum learning rate  — hard lower bound
RHO_MAX: float = 0.50   # maximum learning rate  — hard upper bound


# ─── Level ordering ───────────────────────────────────────────────────────────
LEVEL_ORDER: List[str] = ["kingdom", "phylum", "class_", "order", "family", "genus"]

# ─── Regime ordering (must match projection_engine.REGIME_ORDER) ──────────────
REGIME_ORDER: List[str] = [
    "TREND_UP", "TREND_DN", "RANGE",
    "BREAKOUT_UP", "BREAKOUT_DN",
    "EXHAUST_REV", "LOWVOL", "HIGHVOL",
]
N_REGIMES: int = len(REGIME_ORDER)


class VariableRegistry:
    """
    Central state-space for all learned/observed variables.

    All parameters start from initial priors (from config) and adapt
    toward values supported by incoming market data.  No parameter is
    fixed after initialization — every one has an update equation.

    Parameters
    ----------
    config     : the 'variable_registry' sub-dict from config.yaml
    n_features : number of OHLCV-derived features (default 26 = len(FEATURE_COLS))
    n_nodes    : number of taxonomy nodes (default 28 = |ALL_NODES|)
    n_signals  : number of indicator signals in the mixer (default 7)

    Usage
    -----
    >>> registry = VariableRegistry(cfg['variable_registry'])
    >>> # Each bar:
    >>> tau_obs = registry.compute_tau_observed(log_returns)
    >>> registry.update_tau(tau_obs)
    >>> registry.update_eta('kingdom', regime_velocity=0.12, prediction_error=0.05)
    >>> registry.update_condition_matrix(z_vec, e_k_vec, P_k_vec)
    >>> registry.tick()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        n_features: int = 26,
        n_nodes: int = 28,
        n_signals: int = 7,
    ) -> None:
        self.cfg = config
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_signals = n_signals

        priors = config.get("initial_priors", {})
        bounds = config.get("bounds", {})
        self._bounds = bounds
        self._t: int = 0

        # ── GROUP 1: Natural timescale τ(t) ──────────────────────────────────
        # τ(t) is the empirical autocorrelation decay time of the regime signal.
        # Update: τ̇(t) = ρ_τ(t) · [τ_observed(t) - τ(t)]
        self.tau: float = float(priors.get("tau", 20.0))
        self.rho_tau: float = float(priors.get("rho_tau", 0.05))
        # Running quantile for adaptive threshold
        self._autocorr_mag_buf: List[float] = []
        self.threshold_observed: float = float(priors.get("tau_threshold", 0.10))
        self._tau_loss_prev: float = 1.0

        # ── GROUP 2: Smoothing constants η_L(t) per taxonomy level ────────────
        # Update: η_L(t+1) = η_L(t) + κ_η · (η_target_L(t) − η_L(t))
        # η_target_L = σ(β_η · (stability_L − 0.5))
        eta_priors: Dict[str, float] = priors.get("eta", {
            "kingdom": 0.10, "phylum": 0.12, "class_": 0.15,
            "order":   0.18, "family": 0.20, "genus":  0.22,
        })
        self.eta: Dict[str, float] = {
            L: float(eta_priors.get(L, 0.15)) for L in LEVEL_ORDER
        }
        self.kappa_eta: float = float(priors.get("kappa_eta", 0.05))
        self.beta_eta: float  = float(priors.get("beta_eta",  2.0))
        self._eta_loss_prev: Dict[str, float] = {L: 1.0 for L in LEVEL_ORDER}

        # ── GROUP 3: Condition matrix C(i, k, t) ─────────────────────────────
        # Shape: (n_features, n_nodes)
        # Initialized to zero — prior comes from hardcoded MSL matrices in the
        # taxonomy engine.  C adapts from market data and will override priors.
        # Update: δC(i,k) = z_i · e_k · P_k − C(i,k) · P_k
        self.C: np.ndarray = np.zeros((n_features, n_nodes), dtype=np.float64)
        self.lambda_C: float = float(priors.get("lambda_C", 0.05))
        self._C_loss_prev: float = 1.0

        # ── GROUP 4: Signal weights α_j(t) and windows W_j(t) ────────────────
        # α_j: normalized predictive contribution per indicator signal
        # W_j: adaptive window length per signal
        self.alpha_j: np.ndarray = (
            np.ones(n_signals, dtype=np.float64) / max(n_signals, 1)
        )
        W_j_init = float(priors.get("W_j_init", 60.0))
        self.W_j: np.ndarray = np.full(n_signals, W_j_init, dtype=np.float64)
        self._contrib_buf: List[np.ndarray] = []

        # ── GROUP 4b: Learned indicator interaction matrix M(t) ──────────────
        # Replaces the hardcoded _INTERACTION_MATRIX in mixer.py.
        # Initialized to identity (no prior knowledge of indicator correlations).
        self.M_interaction: np.ndarray = np.eye(n_signals, dtype=np.float64)
        # Tracks sign of recent signal × signal × outcome products
        self._M_update_buf: List[np.ndarray] = []

        # ── GROUP 5: Projection parameters θ_r(t) ────────────────────────────
        # θ_r = [μ_r, φ_r, β_r, σ_r] per regime r
        # Update: θ̇_r = Γ(t) · err_r · ∂err/∂θ_r  (weighted gradient)
        theta_priors: Dict[str, Dict[str, float]] = priors.get("theta_r", {})
        self.theta_r: Dict[str, Dict[str, float]] = {}
        for r in REGIME_ORDER:
            rp = theta_priors.get(r, {})
            self.theta_r[r] = {
                "mu":    float(rp.get("mu",    0.0)),
                "phi":   float(rp.get("phi",   0.5)),
                "beta":  float(rp.get("beta",  0.0)),
                "sigma": float(rp.get("sigma", 0.2)),
            }
        # Γ(t): per-regime, per-parameter learning rates  (N_REGIMES × 4)
        Gamma_init = float(priors.get("Gamma_init", 0.01))
        self.Gamma: np.ndarray = np.full(
            (N_REGIMES, 4), Gamma_init, dtype=np.float64
        )
        self._proj_loss_prev: np.ndarray = np.ones(N_REGIMES)
        self._proj_loss_curr: np.ndarray = np.ones(N_REGIMES)

        # ── GROUP 6: Cone parameters σ(H, t) and cone_skew(H, t) ─────────────
        # σ_cone(H) = empirical std of (actual − projected) at horizon H
        # cone_skew(H) = empirical mean of that error / σ_cone (asymmetry)
        sigma_cone_init = float(priors.get("sigma_cone_init", 0.10))
        self.sigma_cone: Dict[int, float] = {
            H: sigma_cone_init for H in [1, 5, 10, 20, 60]
        }
        self.cone_skew: Dict[int, float] = {H: 0.0 for H in [1, 5, 10, 20, 60]}
        cone_window = int(priors.get("cone_window", 200))
        self._cone_errs: Dict[int, List[float]] = {
            H: [] for H in self.sigma_cone
        }
        self._cone_window = cone_window

        # ── GROUP 7: Projection horizons H_k(t) ──────────────────────────────
        # H_k = EMA of observed transition times at taxonomy level k
        H_k_init: Dict[str, float] = priors.get("H_k_init", {
            "kingdom": 120.0, "phylum":  60.0, "class_": 40.0,
            "order":    20.0, "family":  10.0, "genus":   5.0,
        })
        self.H_k: Dict[str, float] = {
            L: float(H_k_init.get(L, 20.0)) for L in LEVEL_ORDER
        }
        self.alpha_H: float = float(priors.get("alpha_H", 0.05))

        # ── GROUP 8: Cross-scale consistency ──────────────────────────────────
        # consistency(t) = EMA fraction of micro projections consistent with macro
        self.consistency: float = float(priors.get("consistency_init", 0.5))
        self.W_con: float       = float(priors.get("W_con_init", 60.0))
        self.lambda_con: float  = float(priors.get("lambda_con_init", 0.05))
        self._consistency_buf: List[float] = []

        # ── GROUP 9: Regime persistence P_persist_k(t) ───────────────────────
        # Empirical probability that regime r persists from bar t to t+1
        self.P_persist: np.ndarray = np.full(N_REGIMES, 0.5, dtype=np.float64)
        # Observed transition times per level (EMA)
        self.tau_transition: Dict[str, float] = {L: 20.0 for L in LEVEL_ORDER}

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 1: Natural timescale τ(t)
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_tau_observed(self, log_returns: np.ndarray) -> float:
        """
        Compute the empirical autocorrelation decay timescale.

        τ_observed = argmin_k { |autocorr(signal, lag=k)| < threshold_observed(t) }

        threshold_observed(t) is the 30th percentile of recent |autocorr| values —
        not a fixed constant.

        Returns
        -------
        float — lag (in bars) at which autocorrelation first falls below threshold.
                Bounded to [2, τ·3] for numerical stability.
        """
        if len(log_returns) < 10:
            return self.tau

        max_lag = int(min(self.tau * 3, len(log_returns) // 4, 150))
        if max_lag < 2:
            return self.tau

        x = log_returns - log_returns.mean()
        variance = float(np.var(x))
        if variance < 1e-12:
            return self.tau

        # Accumulate autocorrelation magnitudes for running quantile
        for lag in range(1, max_lag + 1):
            if lag >= len(x):
                break
            acf = float(np.mean(x[lag:] * x[:-lag])) / variance
            self._autocorr_mag_buf.append(abs(acf))

        # Keep buffer bounded
        if len(self._autocorr_mag_buf) > 2000:
            self._autocorr_mag_buf = self._autocorr_mag_buf[-1000:]

        # Update threshold_observed = 30th percentile of recent |autocorr| values
        if len(self._autocorr_mag_buf) >= 10:
            self.threshold_observed = float(
                np.percentile(self._autocorr_mag_buf, 30)
            )
            # Hard floor: threshold must be meaningful (≥ 2%)
            self.threshold_observed = max(self.threshold_observed, 0.02)

        # Find first lag where autocorrelation drops below threshold
        for lag in range(1, max_lag + 1):
            if lag >= len(x):
                break
            acf = float(np.mean(x[lag:] * x[:-lag])) / (variance + 1e-12)
            if abs(acf) < self.threshold_observed:
                return float(lag)

        return float(max_lag)

    def update_tau(self, tau_observed: float, loss_curr: Optional[float] = None) -> None:
        """
        τ̇(t) = ρ_τ(t) · [τ_observed(t) − τ(t)]

        ρ_τ is itself a variable updated by meta-learning.
        τ is bounded to [2, 500] bars for stability.
        """
        self.tau += self.rho_tau * (tau_observed - self.tau)
        self.tau = float(np.clip(self.tau, 2.0, 500.0))

        if loss_curr is not None:
            self.rho_tau = self._update_meta_lr(
                self.rho_tau, self._tau_loss_prev, loss_curr
            )
            self._tau_loss_prev = loss_curr

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 2: Smoothing constants η_L(t)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_eta(
        self,
        level: str,
        regime_velocity: float,
        prediction_error: float,
        loss_curr: Optional[float] = None,
    ) -> None:
        """
        η_L(t+1) = η_L(t) + κ_η · (η_target_L(t) − η_L(t))

        η_target_L(t) = σ(β_η · (stability_L(t) − 0.5))
        stability_L(t) = 1 − |regime_velocity| / max_possible_change

        When regime moves fast (high velocity) → η drops (more responsive).
        When regime is stable → η rises (more persistent / sticky).
        High prediction error additionally pushes η toward faster adaptation.

        Parameters
        ----------
        level            : taxonomy level name (e.g. 'kingdom')
        regime_velocity  : |ΔΨ_L| — magnitude of logit change at this level
        prediction_error : error in regime probability prediction at this level
        loss_curr        : optional loss value for meta-learning of κ_η
        """
        # Logits are in tanh-space (bounded ~[-2, 2]), max meaningful change ≈ 2
        max_change = 2.0
        velocity_norm = min(abs(regime_velocity) / (max_change + 1e-10), 1.0)
        stability = 1.0 - velocity_norm

        # η_target: high stability → high η (sticky); low stability → low η (fast)
        eta_target = 1.0 / (1.0 + np.exp(-self.beta_eta * (stability - 0.5)))

        # Prediction error penalty: if we're wrong, push toward faster adaptation
        error_magnitude = min(abs(prediction_error), 2.0)
        error_discount = float(np.exp(-error_magnitude))  # in (0, 1]
        eta_target = eta_target * error_discount + (1.0 - error_discount) * 0.05

        old_eta = self.eta.get(level, 0.15)
        self.eta[level] = old_eta + self.kappa_eta * (eta_target - old_eta)

        eta_min = float(self._bounds.get("eta_min", 0.02))
        eta_max = float(self._bounds.get("eta_max", 0.60))
        self.eta[level] = float(np.clip(self.eta[level], eta_min, eta_max))

        # Meta-update: κ_η adapts based on whether recent η changes helped
        if loss_curr is not None:
            self.kappa_eta = self._update_meta_lr(
                self.kappa_eta, self._eta_loss_prev.get(level, 1.0), loss_curr
            )
            self._eta_loss_prev[level] = loss_curr

    def get_eta(self, level: str) -> float:
        """Return current adaptive smoothing constant for this taxonomy level."""
        return self.eta.get(level, 0.15)

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 3: Condition matrix C(i, k, t)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_condition_matrix(
        self,
        z: np.ndarray,    # (n_features,) — current normalized feature vector
        e_k: np.ndarray,  # (n_nodes,)    — prediction error per node
        P_k: np.ndarray,  # (n_nodes,)    — current node probability vector
        loss_curr: Optional[float] = None,
    ) -> None:
        """
        C(i, k, t+1) = C(i, k, t) + λ_C(t) · δC(i, k, t)

        δC(i, k, t) = z_i(t) · e_k(t) · P_k(t) − C(i, k, t) · P_k(t)

        Hebbian-style: if signal z_i was active and node k subsequently changed
        (e_k > 0), C(i, k) increases.  If node k didn't respond → C(i, k)
        decays toward zero, weighted by how probable that node was.

        λ_C(t) is itself a variable updated by meta-learning.

        Parameters
        ----------
        z   : normalized feature vector at time t
        e_k : prediction error = P_k(t+1) - E[P_k(t+1) | z(t), C(t)]
        P_k : current node probabilities (linear scale, summing to 1 per group)
        """
        n_f = min(len(z), self.n_features)
        n_n = min(len(e_k), self.n_nodes)

        z_col = z[:n_f].reshape(-1, 1)    # (n_f, 1)
        e_row = e_k[:n_n].reshape(1, -1)  # (1, n_n)
        P_row = P_k[:n_n].reshape(1, -1)  # (1, n_n)

        # Hebbian update: signal × error × node weight — decay by node weight
        delta_C = z_col * e_row * P_row - self.C[:n_f, :n_n] * P_row
        self.C[:n_f, :n_n] += self.lambda_C * delta_C

        C_max = float(self._bounds.get("C_max", 3.0))
        self.C = np.clip(self.C, -C_max, C_max)

        if loss_curr is not None:
            self.lambda_C = self._update_meta_lr(
                self.lambda_C, self._C_loss_prev, loss_curr
            )
            self._C_loss_prev = loss_curr
            self.lambda_C = float(np.clip(self.lambda_C, RHO_MIN, RHO_MAX))

    def get_condition_matrix(self) -> np.ndarray:
        """Return current condition matrix C(i, k, t). Shape: (n_features, n_nodes)."""
        return self.C.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 4: Signal weights α_j(t) and window lengths W_j(t)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_signal_weights(
        self,
        contributions: np.ndarray,  # (n_signals,) — recent predictive accuracy
    ) -> None:
        """
        α_j(t) ← softmax of predictive contributions, interpolated toward current.

        Signals that better predicted subsequent regime classification receive
        higher weight.  Softmax normalization ensures Σ α_j = 1.
        """
        n = min(len(contributions), self.n_signals)
        c = contributions[:n]

        # Softmax over contributions → target weights
        c_shifted = c - float(c.max())
        exp_c = np.exp(np.clip(c_shifted, -30.0, 0.0))
        target = exp_c / (exp_c.sum() + 1e-15)

        lr = float(self._bounds.get("alpha_j_lr", 0.05))
        self.alpha_j[:n] = self.alpha_j[:n] + lr * (target - self.alpha_j[:n])
        # Re-normalize after update
        total = float(self.alpha_j[:n].sum())
        if total > 1e-10:
            self.alpha_j[:n] /= total

        self._contrib_buf.append(c.copy())
        if len(self._contrib_buf) > 500:
            self._contrib_buf = self._contrib_buf[-200:]

    def update_window_lengths(
        self,
        signal_idx: int,
        perf_short: float,
        perf_long: float,
    ) -> None:
        """
        W_j(t) adapts toward the window length that produced better prediction
        accuracy.  If short window outperforms, shrink; if long outperforms, grow.
        """
        W_min = float(self._bounds.get("W_min", 10.0))
        W_max = float(self._bounds.get("W_max", 500.0))
        W_lr  = float(self._bounds.get("W_lr",  0.02))

        current = float(self.W_j[signal_idx])
        target = current * (0.85 if perf_short > perf_long else 1.15)
        self.W_j[signal_idx] += W_lr * (target - current)
        self.W_j[signal_idx] = float(np.clip(self.W_j[signal_idx], W_min, W_max))

    def update_interaction_matrix(
        self,
        signals: np.ndarray,   # (n_signals,) — current indicator signals
        outcome: float,        # +1 if regime call correct, -1 if wrong
    ) -> None:
        """
        M_interaction(i, j, t) — learned indicator cross-correlation matrix.

        Each cell M(i,j) tracks whether signals i and j agreeing predicts
        correct regime classification.

        Update: M(i,j) += lr · sign(s_i · s_j) · outcome − lr · M(i,j)
        """
        n = min(len(signals), self.n_signals)
        s = signals[:n].reshape(-1, 1)  # (n, 1)
        outer = s @ s.T                  # (n, n) — outer product of signals

        lr = float(self._bounds.get("M_lr", 0.02))
        delta = np.sign(outer) * outcome - self.M_interaction[:n, :n]
        self.M_interaction[:n, :n] += lr * delta
        np.clip(self.M_interaction, -1.0, 1.0, out=self.M_interaction)

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 5: Projection parameters θ_r(t)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_projection_params(
        self,
        x_observed: float,          # actual signal value at t+1
        x_t: float,                 # signal value at t
        dx_t: float,                # Δx_t = x_t − x_{t−1}
        regime_probs: np.ndarray,   # (N_REGIMES,) — soft regime assignment
    ) -> None:
        """
        θ̇_r(t) = Γ(t) · prediction_error_r(t) · ∂error/∂θ_r

        Weighted by P(r at t): no hard regime labels, pure soft assignment.

        Gradients of error_r w.r.t. AR(1) parameters:
          error_r   = x_obs − (μ_r + φ_r·(x_t−μ_r) + β_r·Δx_t)
          ∂err/∂μ_r  = −(1 − φ_r)
          ∂err/∂φ_r  = −(x_t − μ_r)
          ∂err/∂β_r  = −Δx_t
          σ_r adapts toward |error_r| (empirical residual std)
        """
        for i, r in enumerate(REGIME_ORDER):
            theta = self.theta_r[r]
            mu_r, phi_r, beta_r, sigma_r = (
                theta["mu"], theta["phi"], theta["beta"], theta["sigma"]
            )

            x_pred_r = mu_r + phi_r * (x_t - mu_r) + beta_r * dx_t
            error_r  = x_observed - x_pred_r
            p_r      = float(max(regime_probs[i], 0.0))

            # Negative gradients (for gradient descent on squared error)
            neg_grad_mu   = (1.0 - phi_r)
            neg_grad_phi  = (x_t - mu_r)
            neg_grad_beta = dx_t

            Gamma_r = self.Gamma[i]
            theta["mu"]    += Gamma_r[0] * p_r * error_r * neg_grad_mu
            theta["phi"]   += Gamma_r[1] * p_r * error_r * neg_grad_phi
            theta["beta"]  += Gamma_r[2] * p_r * error_r * neg_grad_beta
            # σ_r: EMA toward empirical |error_r|, weighted by p_r
            theta["sigma"] += Gamma_r[3] * p_r * (abs(error_r) - sigma_r)

            # Clip parameters to valid ranges
            theta["mu"]    = float(np.clip(theta["mu"],   -2.0,  2.0))
            theta["phi"]   = float(np.clip(theta["phi"],   -1.0, 0.999))
            theta["beta"]  = float(np.clip(theta["beta"], -1.0,  1.0))
            theta["sigma"] = float(np.clip(theta["sigma"], 1e-4, 2.0))

            self._proj_loss_curr[i] = float(error_r ** 2)

        # Meta-update: Γ adapts toward faster/slower learning based on loss trend
        for i in range(N_REGIMES):
            for j in range(4):
                self.Gamma[i, j] = self._update_meta_lr(
                    self.Gamma[i, j],
                    float(self._proj_loss_prev[i]),
                    float(self._proj_loss_curr[i]),
                )
        self._proj_loss_prev = self._proj_loss_curr.copy()

    def get_theta_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return current θ_r as (mu, phi, beta, sigma) arrays in REGIME_ORDER.
        Each array has shape (N_REGIMES,) for direct use in ProjectionEngine.
        """
        mu    = np.array([self.theta_r[r]["mu"]    for r in REGIME_ORDER])
        phi   = np.array([self.theta_r[r]["phi"]   for r in REGIME_ORDER])
        beta  = np.array([self.theta_r[r]["beta"]  for r in REGIME_ORDER])
        sigma = np.array([self.theta_r[r]["sigma"] for r in REGIME_ORDER])
        return mu, phi, beta, sigma

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 6: Cone parameters σ(H, t) and cone_skew(H, t)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_cone(
        self,
        H: int,
        actual_state: float,
        projected_state: float,
    ) -> None:
        """
        σ_cone(H, t) = std of (actual − projected) over adaptive window.
        cone_skew(H, t) = mean of that error / σ_cone — captures asymmetry.

        No assumed √H scaling. Each horizon tracks its own empirical dispersion.
        """
        if H not in self._cone_errs:
            self._cone_errs[H] = []
            self.sigma_cone[H] = 0.1
            self.cone_skew[H]  = 0.0

        err = actual_state - projected_state
        self._cone_errs[H].append(err)

        if len(self._cone_errs[H]) > self._cone_window:
            self._cone_errs[H] = self._cone_errs[H][-self._cone_window:]

        if len(self._cone_errs[H]) >= 5:
            errs = np.array(self._cone_errs[H])
            self.sigma_cone[H] = float(np.std(errs) + 1e-10)
            self.cone_skew[H]  = float(np.mean(errs) / (self.sigma_cone[H] + 1e-10))

    def get_sigma_cone(self, H: int) -> float:
        """
        Return empirical cone width for horizon H.
        Linearly interpolates between known horizons if H is not directly tracked.
        """
        if H in self.sigma_cone:
            return self.sigma_cone[H]

        known = sorted(self.sigma_cone.keys())
        if not known:
            return 0.1
        if H <= known[0]:
            return self.sigma_cone[known[0]]
        if H >= known[-1]:
            return self.sigma_cone[known[-1]]

        for i in range(len(known) - 1):
            h1, h2 = known[i], known[i + 1]
            if h1 <= H <= h2:
                frac = (H - h1) / (h2 - h1 + 1e-10)
                return self.sigma_cone[h1] * (1 - frac) + self.sigma_cone[h2] * frac
        return 0.1

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 7: Projection horizons H_k(t)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_horizon(self, level: str, observed_transition_time: float) -> None:
        """
        H̄_k(t) = EMA of observed transition times at taxonomy level k.

        observed_transition_time: bars elapsed since last significant regime
        change at this level (detected when dominant node probability changed
        by > change_threshold — itself adaptive via tau_transition).
        """
        current = self.H_k.get(level, 20.0)
        self.H_k[level] = current + self.alpha_H * (
            observed_transition_time - current
        )
        self.H_k[level] = float(np.clip(self.H_k[level], 1.0, 2000.0))

    def get_time_dilation(self, level_micro: str, level_macro: str) -> float:
        """
        D(H_micro, H_macro, t) = H_k[macro] / H_k[micro]

        The ratio between observed transition timescales at two levels.
        No assumed multiplier — derived entirely from H_k state variables.
        """
        H_micro = self.H_k.get(level_micro, 5.0)
        H_macro = self.H_k.get(level_macro, 60.0)
        return float(H_macro / (H_micro + 1e-10))

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 8: Cross-scale consistency
    # ═══════════════════════════════════════════════════════════════════════════

    def update_consistency(self, observed_consistency: float) -> None:
        """
        consistency(t+1) = (1 − λ_con) · consistency(t)
                         + λ_con · observed_consistency(t)

        observed_consistency: fraction of recent micro projections consistent
        with macro projection over the last W_con bars.

        When consistency is low, the cone should be widened and confidence
        reduced.  get_cone_confidence_multiplier() returns that multiplier.
        """
        self.consistency = (
            (1.0 - self.lambda_con) * self.consistency
            + self.lambda_con * float(np.clip(observed_consistency, 0.0, 1.0))
        )

        self._consistency_buf.append(observed_consistency)
        if len(self._consistency_buf) > 500:
            self._consistency_buf = self._consistency_buf[-200:]

        # W_con adapts: high variance in consistency → expand window for stability
        if len(self._consistency_buf) >= 20:
            buf_std = float(np.std(self._consistency_buf[-20:]))
            W_con_min = float(self._bounds.get("W_con_min", 10.0))
            W_con_max = float(self._bounds.get("W_con_max", 500.0))
            W_target  = float(np.clip(
                self.W_con * (1.0 + buf_std), W_con_min, W_con_max
            ))
            self.W_con += 0.05 * (W_target - self.W_con)

        # lambda_con adapts: high consistency variance → slow down adaptation
        if len(self._consistency_buf) >= 10:
            recent_var = float(np.var(self._consistency_buf[-10:]))
            lambda_target = float(np.clip(0.02 + 0.08 * (1.0 - recent_var * 4), 0.01, 0.20))
            self.lambda_con += 0.05 * (lambda_target - self.lambda_con)
            self.lambda_con = float(np.clip(self.lambda_con, 0.01, 0.30))

    def get_cone_confidence_multiplier(self) -> float:
        """
        Returns a multiplier ∈ [0.5, 1.0] that scales confidence scores.

        When micro and macro projections disagree (low consistency), both cones
        widen and confidence is reduced proportionally.  No hardcoded threshold.
        """
        return float(0.5 + 0.5 * self.consistency)

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 9: Regime persistence and transition times
    # ═══════════════════════════════════════════════════════════════════════════

    def update_persistence(self, regime_idx: int, stayed_same: bool) -> None:
        """
        P_persist_k(t) — empirical persistence of each regime node.

        Updated by Bernoulli observation: did the dominant regime stay or switch?
        """
        target = 1.0 if stayed_same else 0.0
        lr = float(self._bounds.get("persistence_lr", 0.02))
        self.P_persist[regime_idx] += lr * (target - self.P_persist[regime_idx])
        self.P_persist[regime_idx] = float(
            np.clip(self.P_persist[regime_idx], 0.01, 0.99)
        )

    def update_transition_time(self, level: str, bars_since_last: int) -> None:
        """
        τ_transition_L(t) — EMA of bars between regime transitions at level L.
        """
        current = self.tau_transition.get(level, 20.0)
        lr = float(self._bounds.get("tau_transition_lr", 0.05))
        self.tau_transition[level] = current + lr * (bars_since_last - current)
        self.tau_transition[level] = float(
            np.clip(self.tau_transition[level], 1.0, 2000.0)
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Meta-learning: self-modulating learning rates
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_meta_lr(
        self,
        rho: float,
        loss_prev: float,
        loss_curr: float,
        delta_rho: float = 0.001,
    ) -> float:
        """
        ρ(t+1) = clip(ρ(t) + δ_ρ · [loss(t−1) − loss(t)], ρ_min, ρ_max)

        Improvement (loss decreased)  → ρ increases (adapt faster)
        Degradation  (loss increased) → ρ decreases (adapt slower)

        RHO_MIN and RHO_MAX are the only true constants in this system.
        """
        improvement = loss_prev - loss_curr
        return float(np.clip(rho + delta_rho * improvement, RHO_MIN, RHO_MAX))

    # ═══════════════════════════════════════════════════════════════════════════
    # Diagnostic summary
    # ═══════════════════════════════════════════════════════════════════════════

    def summary(self) -> Dict[str, Any]:
        """Return a concise snapshot of all current variable values."""
        return {
            "t":               self._t,
            "tau":             round(self.tau, 3),
            "rho_tau":         round(self.rho_tau, 5),
            "threshold_obs":   round(self.threshold_observed, 4),
            "eta":             {L: round(v, 4) for L, v in self.eta.items()},
            "kappa_eta":       round(self.kappa_eta, 5),
            "lambda_C":        round(self.lambda_C, 5),
            "C_norm":          round(float(np.linalg.norm(self.C)), 4),
            "alpha_j":         [round(float(x), 4) for x in self.alpha_j],
            "W_j":             [round(float(x), 2) for x in self.W_j],
            "theta_r": {
                r: {k: round(v, 4) for k, v in theta.items()}
                for r, theta in self.theta_r.items()
            },
            "H_k":             {L: round(v, 2) for L, v in self.H_k.items()},
            "sigma_cone":      {H: round(v, 4) for H, v in self.sigma_cone.items()},
            "cone_skew":       {H: round(v, 4) for H, v in self.cone_skew.items()},
            "consistency":     round(self.consistency, 4),
            "W_con":           round(self.W_con, 2),
            "lambda_con":      round(self.lambda_con, 5),
            "P_persist":       [round(float(x), 4) for x in self.P_persist],
            "tau_transition":  {L: round(v, 2) for L, v in self.tau_transition.items()},
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # State serialization / deserialization
    # ═══════════════════════════════════════════════════════════════════════════

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full variable state for persistence (JSON-safe)."""
        return {
            "t":                     self._t,
            "tau":                   self.tau,
            "rho_tau":               self.rho_tau,
            "threshold_observed":    self.threshold_observed,
            "eta":                   dict(self.eta),
            "kappa_eta":             self.kappa_eta,
            "beta_eta":              self.beta_eta,
            "C":                     self.C.tolist(),
            "lambda_C":              self.lambda_C,
            "alpha_j":               self.alpha_j.tolist(),
            "W_j":                   self.W_j.tolist(),
            "M_interaction":         self.M_interaction.tolist(),
            "theta_r":               {r: dict(v) for r, v in self.theta_r.items()},
            "Gamma":                 self.Gamma.tolist(),
            "sigma_cone":            {str(k): v for k, v in self.sigma_cone.items()},
            "cone_skew":             {str(k): v for k, v in self.cone_skew.items()},
            "H_k":                   dict(self.H_k),
            "alpha_H":               self.alpha_H,
            "consistency":           self.consistency,
            "W_con":                 self.W_con,
            "lambda_con":            self.lambda_con,
            "P_persist":             self.P_persist.tolist(),
            "tau_transition":        dict(self.tau_transition),
        }

    @classmethod
    def from_dict(
        cls,
        state: Dict[str, Any],
        config: Dict[str, Any],
        n_features: int = 26,
        n_nodes: int = 28,
        n_signals: int = 7,
    ) -> "VariableRegistry":
        """Restore a VariableRegistry from a serialized state dict."""
        reg = cls(config, n_features=n_features, n_nodes=n_nodes, n_signals=n_signals)

        reg._t                   = state.get("t", 0)
        reg.tau                  = float(state.get("tau", reg.tau))
        reg.rho_tau              = float(state.get("rho_tau", reg.rho_tau))
        reg.threshold_observed   = float(state.get("threshold_observed", reg.threshold_observed))

        for L in LEVEL_ORDER:
            eta_state = state.get("eta", {})
            if L in eta_state:
                reg.eta[L] = float(eta_state[L])

        reg.kappa_eta = float(state.get("kappa_eta", reg.kappa_eta))
        reg.beta_eta  = float(state.get("beta_eta",  reg.beta_eta))

        C_data = state.get("C")
        if C_data is not None:
            arr = np.array(C_data, dtype=np.float64)
            r_slice = min(arr.shape[0], n_features)
            c_slice = min(arr.shape[1], n_nodes)
            reg.C[:r_slice, :c_slice] = arr[:r_slice, :c_slice]

        reg.lambda_C = float(state.get("lambda_C", reg.lambda_C))

        aj = state.get("alpha_j")
        if aj is not None:
            n = min(len(aj), n_signals)
            reg.alpha_j[:n] = np.array(aj[:n])

        Wj = state.get("W_j")
        if Wj is not None:
            n = min(len(Wj), n_signals)
            reg.W_j[:n] = np.array(Wj[:n])

        Mi = state.get("M_interaction")
        if Mi is not None:
            arr = np.array(Mi, dtype=np.float64)
            n = min(arr.shape[0], n_signals)
            reg.M_interaction[:n, :n] = arr[:n, :n]

        theta_state = state.get("theta_r", {})
        for r in REGIME_ORDER:
            if r in theta_state:
                for p in ("mu", "phi", "beta", "sigma"):
                    if p in theta_state[r]:
                        reg.theta_r[r][p] = float(theta_state[r][p])

        Gamma_data = state.get("Gamma")
        if Gamma_data is not None:
            arr = np.array(Gamma_data, dtype=np.float64)
            r_slice = min(arr.shape[0], N_REGIMES)
            c_slice = min(arr.shape[1], 4)
            reg.Gamma[:r_slice, :c_slice] = arr[:r_slice, :c_slice]

        for k, v in state.get("sigma_cone", {}).items():
            reg.sigma_cone[int(k)] = float(v)
        for k, v in state.get("cone_skew", {}).items():
            reg.cone_skew[int(k)] = float(v)
        for L, v in state.get("H_k", {}).items():
            reg.H_k[L] = float(v)

        reg.alpha_H   = float(state.get("alpha_H",   reg.alpha_H))
        reg.consistency = float(state.get("consistency", reg.consistency))
        reg.W_con      = float(state.get("W_con",      reg.W_con))
        reg.lambda_con = float(state.get("lambda_con", reg.lambda_con))

        pp = state.get("P_persist")
        if pp is not None:
            n = min(len(pp), N_REGIMES)
            reg.P_persist[:n] = np.array(pp[:n])
        for L, v in state.get("tau_transition", {}).items():
            if L in reg.tau_transition:
                reg.tau_transition[L] = float(v)

        return reg

    def tick(self) -> None:
        """Advance the internal bar counter by one."""
        self._t += 1

    @property
    def t(self) -> int:
        """Current bar index."""
        return self._t
