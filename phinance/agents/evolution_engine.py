"""
phinance.agents.evolution_engine
=====================================

EvolutionEngine — continuously evolves trading strategies without
human intervention using a genetic/evolutionary feedback loop.

Architecture
------------
The engine runs an **evolution loop** with these phases each generation:

  1. **Population init**  — seed with proposals from StrategyProposerAgent.
  2. **Evaluate**         — run vectorized backtest on every individual.
  3. **Select**           — tournament-select survivors (top-k by fitness).
  4. **Mutate**           — perturb weights + swap one indicator.
  5. **Crossover**        — blend weights from two parents.
  6. **Validate & Deploy**— best individual → StrategyValidator → AutonomousDeployer.
  7. **Archive**          — store every generation in evolution history.

Fitness function
----------------
  fitness = Sharpe × (1 − max_drawdown) × √num_trades

  This rewards risk-adjusted returns while penalising strategies that
  barely trade.

Public API
----------
  Individual             — one candidate strategy (weights, indicators, stats)
  GenerationResult       — summary of one generation
  EvolutionConfig        — tunable hyper-parameters
  EvolutionEngine        — main controller
  run_evolution          — convenience one-shot function
"""

from __future__ import annotations

import copy
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phinance.agents.strategy_proposer import StrategyProposerAgent, StrategyProposal
from phinance.agents.strategy_validator import StrategyValidator, ValidationResult
from phinance.agents.autonomous_deployer import (
    AutonomousDeployer,
    DeploymentRecord,
    StrategyRegistry,
)
from phinance.backtest.vectorized import run_vectorized_backtest
from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Individual:
    """
    One candidate strategy in the evolutionary population.

    Attributes
    ----------
    individual_id  : str   — unique UUID
    indicators     : list  — list of indicator names
    weights        : dict  — {name: float} normalised to sum 1.0
    blend_method   : str   — blending method label
    fitness        : float — computed fitness score (higher is better)
    sharpe         : float
    max_drawdown   : float
    win_rate       : float
    num_trades     : int
    total_return   : float
    generation     : int   — which generation produced this individual
    """

    individual_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    indicators:    List[str] = field(default_factory=list)
    weights:       Dict[str, float] = field(default_factory=dict)
    blend_method:  str = "weighted_sum"
    fitness:       float = 0.0
    sharpe:        float = 0.0
    max_drawdown:  float = 0.0
    win_rate:      float = 0.0
    num_trades:    int = 0
    total_return:  float = 0.0
    generation:    int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "individual_id": self.individual_id,
            "indicators":    self.indicators,
            "weights":       self.weights,
            "blend_method":  self.blend_method,
            "fitness":       self.fitness,
            "sharpe":        self.sharpe,
            "max_drawdown":  self.max_drawdown,
            "win_rate":      self.win_rate,
            "num_trades":    self.num_trades,
            "total_return":  self.total_return,
            "generation":    self.generation,
        }

    def __repr__(self) -> str:
        return (
            f"Individual(id={self.individual_id}, "
            f"fitness={self.fitness:.3f}, "
            f"inds={self.indicators})"
        )


@dataclass
class GenerationResult:
    """
    Summary of one generation of the evolutionary loop.

    Attributes
    ----------
    generation      : int
    population_size : int
    best_fitness    : float
    mean_fitness    : float
    best_individual : Individual
    deployed        : bool
    deployment_id   : str or None
    elapsed_ms      : float
    """

    generation:      int
    population_size: int
    best_fitness:    float
    mean_fitness:    float
    best_individual: Optional[Individual] = None
    deployed:        bool = False
    deployment_id:   Optional[str] = None
    elapsed_ms:      float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation":      self.generation,
            "population_size": self.population_size,
            "best_fitness":    self.best_fitness,
            "mean_fitness":    self.mean_fitness,
            "deployed":        self.deployed,
            "deployment_id":   self.deployment_id,
            "elapsed_ms":      self.elapsed_ms,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
        }

    def __repr__(self) -> str:
        return (
            f"GenerationResult(gen={self.generation}, "
            f"best={self.best_fitness:.3f}, "
            f"mean={self.mean_fitness:.3f}, "
            f"deployed={self.deployed})"
        )


@dataclass
class EvolutionConfig:
    """
    Hyper-parameters for the EvolutionEngine.

    Attributes
    ----------
    population_size    : int   — individuals per generation (default 10)
    num_generations    : int   — generations to run (default 5)
    tournament_k       : int   — tournament selection size (default 3)
    mutation_rate      : float — probability of weight mutation (default 0.3)
    mutation_strength  : float — Gaussian std for weight perturbation (default 0.1)
    crossover_rate     : float — probability of crossover vs copy (default 0.5)
    min_indicators     : int   — minimum indicators per individual (default 2)
    max_indicators     : int   — maximum indicators per individual (default 5)
    deploy_threshold   : float — minimum fitness to attempt deployment (default 0.1)
    dry_run            : bool  — if True never actually deploy (default True)
    random_seed        : int or None
    """

    population_size:   int   = 10
    num_generations:   int   = 5
    tournament_k:      int   = 3
    mutation_rate:     float = 0.3
    mutation_strength: float = 0.1
    crossover_rate:    float = 0.5
    min_indicators:    int   = 2
    max_indicators:    int   = 5
    deploy_threshold:  float = 0.1
    dry_run:           bool  = True
    random_seed:       Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Fitness function
# ─────────────────────────────────────────────────────────────────────────────


def _fitness(sharpe: float, max_drawdown: float, num_trades: int) -> float:
    """
    fitness = Sharpe × (1 − max_drawdown) × √(num_trades + 1)

    Rewards risk-adjusted returns while requiring meaningful trade activity.
    Returns 0.0 if any component is degenerate.
    """
    if sharpe <= 0 or max_drawdown >= 1.0:
        return 0.0
    return float(sharpe * (1.0 - max_drawdown) * (num_trades + 1) ** 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# EvolutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class EvolutionEngine:
    """
    Continuously evolves trading strategies using a genetic algorithm loop.

    Usage
    -----
    ::

        from phinance.agents.evolution_engine import EvolutionEngine, EvolutionConfig

        engine = EvolutionEngine(
            ohlcv=df,
            config=EvolutionConfig(population_size=8, num_generations=3, dry_run=True),
        )
        history = engine.run()
        best = engine.best_individual
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        config: Optional[EvolutionConfig] = None,
        deployer: Optional[AutonomousDeployer] = None,
        registry: Optional[StrategyRegistry] = None,
    ) -> None:
        self.ohlcv   = ohlcv
        self.config  = config or EvolutionConfig()
        self._rng    = random.Random(self.config.random_seed)
        self._np_rng = np.random.default_rng(self.config.random_seed)

        registry         = registry or StrategyRegistry()
        self._deployer   = deployer or AutonomousDeployer(
            registry=registry,
            dry_run=self.config.dry_run,
        )
        self._validator  = StrategyValidator(min_trades=1)
        self._proposer   = StrategyProposerAgent()

        self.history:        List[GenerationResult] = []
        self.best_individual: Optional[Individual]  = None
        self._all_names      = list(INDICATOR_CATALOG.keys())

    # ── public ───────────────────────────────────────────────────────────────

    def run(self) -> List[GenerationResult]:
        """
        Run the full evolutionary loop.

        Returns
        -------
        list[GenerationResult]
            One entry per generation.
        """
        cfg = self.config
        population = self._init_population(cfg.population_size)

        for gen in range(cfg.num_generations):
            t0 = time.perf_counter()
            population = self._evaluate(population, gen)
            population.sort(key=lambda x: x.fitness, reverse=True)

            best    = population[0]
            mean_f  = float(np.mean([i.fitness for i in population]))

            if (
                self.best_individual is None
                or best.fitness > self.best_individual.fitness
            ):
                self.best_individual = copy.deepcopy(best)

            # Attempt deployment of best individual
            deployed      = False
            deployment_id = None
            if best.fitness >= cfg.deploy_threshold:
                try:
                    rec = self._deploy_individual(best, gen)
                    deployed      = True
                    deployment_id = rec.deployment_id
                    logger.info(
                        "Gen %d: deployed %s (fitness=%.3f)",
                        gen, rec.deployment_id, best.fitness,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Gen %d: deploy failed — %s", gen, exc)

            elapsed = (time.perf_counter() - t0) * 1000.0
            gr = GenerationResult(
                generation=gen,
                population_size=len(population),
                best_fitness=best.fitness,
                mean_fitness=mean_f,
                best_individual=copy.deepcopy(best),
                deployed=deployed,
                deployment_id=deployment_id,
                elapsed_ms=elapsed,
            )
            self.history.append(gr)
            logger.info(repr(gr))

            # Evolve next generation (skip after last)
            if gen < cfg.num_generations - 1:
                population = self._evolve(population)

        return self.history

    def run_once(self, generation: int = 0) -> GenerationResult:
        """Run a single generation and return its result."""
        population = self._init_population(self.config.population_size)
        population = self._evaluate(population, generation)
        population.sort(key=lambda x: x.fitness, reverse=True)

        best   = population[0]
        mean_f = float(np.mean([i.fitness for i in population]))

        if self.best_individual is None or best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(best)

        deployed      = False
        deployment_id = None
        if best.fitness >= self.config.deploy_threshold:
            try:
                rec           = self._deploy_individual(best, generation)
                deployed      = True
                deployment_id = rec.deployment_id
            except Exception:  # noqa: BLE001
                pass

        gr = GenerationResult(
            generation=generation,
            population_size=len(population),
            best_fitness=best.fitness,
            mean_fitness=mean_f,
            best_individual=copy.deepcopy(best),
            deployed=deployed,
            deployment_id=deployment_id,
        )
        self.history.append(gr)
        return gr

    @property
    def evolution_summary(self) -> Dict[str, Any]:
        """Return a summary dict of the full evolution run."""
        if not self.history:
            return {"generations": 0, "best_fitness": 0.0}
        fitnesses = [g.best_fitness for g in self.history]
        return {
            "generations":       len(self.history),
            "best_fitness":      max(fitnesses),
            "final_mean_fitness": self.history[-1].mean_fitness,
            "total_deployments": sum(1 for g in self.history if g.deployed),
            "best_individual":   self.best_individual.to_dict() if self.best_individual else None,
        }

    # ── internal ─────────────────────────────────────────────────────────────

    def _init_population(self, size: int) -> List[Individual]:
        """Seed the population with random individuals."""
        pop = []
        for _ in range(size):
            n = self._rng.randint(
                self.config.min_indicators,
                self.config.max_indicators,
            )
            names = self._rng.sample(self._all_names, min(n, len(self._all_names)))
            weights = self._random_weights(names)
            pop.append(Individual(indicators=names, weights=weights))
        return pop

    def _evaluate(self, population: List[Individual], gen: int) -> List[Individual]:
        """Evaluate each individual via vectorized backtest."""
        closes = self.ohlcv["close"].values.astype(float)
        for ind in population:
            ind.generation = gen
            try:
                blended = self._blend_signals(ind)
                result  = run_vectorized_backtest(
                    symbol="EVAL",
                    signal=blended,
                    closes=closes,
                )
                ind.sharpe        = result.sharpe
                ind.max_drawdown  = result.max_drawdown
                ind.win_rate      = result.win_rate
                ind.num_trades    = result.num_trades
                ind.total_return  = result.total_return
                ind.fitness       = _fitness(result.sharpe, result.max_drawdown, result.num_trades)
            except Exception:  # noqa: BLE001
                ind.fitness = 0.0
        return population

    def _blend_signals(self, ind: Individual) -> np.ndarray:
        """Compute and blend signals for one individual."""
        signals = []
        weights = []
        for name in ind.indicators:
            try:
                sig = compute_indicator(name, self.ohlcv, {})
                if sig is not None and not sig.isna().all():
                    arr = sig.fillna(0.0).values.astype(float)
                    signals.append(arr)
                    weights.append(ind.weights.get(name, 1.0))
            except Exception:  # noqa: BLE001
                pass

        if not signals:
            return np.zeros(len(self.ohlcv))

        total_w = sum(weights) or 1.0
        blended = sum(s * w / total_w for s, w in zip(signals, weights))
        # Clip to [-1, 1]
        return np.clip(blended, -1.0, 1.0)

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Select one individual via tournament selection."""
        k = min(self.config.tournament_k, len(population))
        contestants = self._rng.sample(population, k)
        return max(contestants, key=lambda x: x.fitness)

    def _mutate(self, ind: Individual) -> Individual:
        """Apply mutation to weights and optionally swap an indicator."""
        child = copy.deepcopy(ind)
        child.individual_id = str(uuid.uuid4())[:8]

        # Weight mutation
        for name in child.indicators:
            if self._rng.random() < self.config.mutation_rate:
                delta = float(self._np_rng.normal(0, self.config.mutation_strength))
                child.weights[name] = max(0.01, child.weights.get(name, 0.1) + delta)

        # Normalise weights
        total = sum(child.weights.values()) or 1.0
        child.weights = {k: v / total for k, v in child.weights.items()}

        # Indicator swap (with mutation_rate probability)
        if (
            self._rng.random() < self.config.mutation_rate
            and len(self._all_names) > len(child.indicators)
        ):
            available = [n for n in self._all_names if n not in child.indicators]
            if available:
                old = self._rng.choice(child.indicators)
                new = self._rng.choice(available)
                idx = child.indicators.index(old)
                child.indicators[idx] = new
                child.weights[new]    = child.weights.pop(old, 0.1)

        return child

    def _crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        """Blend weights from two parents; inherit indicator list from parent_a."""
        child = copy.deepcopy(parent_a)
        child.individual_id = str(uuid.uuid4())[:8]

        for name in child.indicators:
            wa = parent_a.weights.get(name, 0.1)
            wb = parent_b.weights.get(name, 0.1)
            alpha = self._rng.random()
            child.weights[name] = alpha * wa + (1 - alpha) * wb

        total = sum(child.weights.values()) or 1.0
        child.weights = {k: v / total for k, v in child.weights.items()}
        return child

    def _evolve(self, population: List[Individual]) -> List[Individual]:
        """Produce next generation via selection + mutation + crossover."""
        cfg    = self.config
        size   = cfg.population_size
        # Elitism: carry best 2 forward unchanged
        elite  = population[:2]
        new_pop: List[Individual] = list(elite)

        while len(new_pop) < size:
            if self._rng.random() < cfg.crossover_rate and len(population) >= 2:
                pa = self._tournament_select(population)
                pb = self._tournament_select(population)
                child = self._crossover(pa, pb)
            else:
                parent = self._tournament_select(population)
                child  = self._mutate(parent)
            new_pop.append(child)

        return new_pop[:size]

    def _deploy_individual(self, ind: Individual, gen: int) -> DeploymentRecord:
        """Validate and deploy the best individual."""
        # Build a minimal proposal for the deployer
        proposal = StrategyProposal(
            indicators={n: {"enabled": True, "params": {}} for n in ind.indicators},
            weights=ind.weights,
            blend_method=ind.blend_method,
            regime="UNKNOWN",
            scores={n: ind.weights.get(n, 0.1) for n in ind.indicators},
            rationale=f"Evolution gen={gen} fitness={ind.fitness:.3f}",
        )
        validation = self._validator.validate(proposal, self.ohlcv)
        record = self._deployer.deploy(
            proposal=proposal,
            validation=validation,
            notes=f"EvolutionEngine gen={gen}",
        )
        return record

    def _random_weights(self, names: List[str]) -> Dict[str, float]:
        """Return normalised random weights for a list of indicator names."""
        raw = {n: self._rng.random() for n in names}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def run_evolution(
    ohlcv: pd.DataFrame,
    population_size: int = 8,
    num_generations: int = 3,
    dry_run: bool = True,
    random_seed: Optional[int] = 42,
) -> Tuple[List[GenerationResult], Optional[Individual]]:
    """
    One-shot evolution run.

    Parameters
    ----------
    ohlcv           : pd.DataFrame — OHLCV data
    population_size : int
    num_generations : int
    dry_run         : bool — never actually deploy if True
    random_seed     : int or None

    Returns
    -------
    (history, best_individual)
    """
    cfg    = EvolutionConfig(
        population_size=population_size,
        num_generations=num_generations,
        dry_run=dry_run,
        random_seed=random_seed,
    )
    engine = EvolutionEngine(ohlcv=ohlcv, config=cfg)
    history = engine.run()
    return history, engine.best_individual
