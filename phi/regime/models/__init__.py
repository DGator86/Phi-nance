"""Concrete regime detector implementations."""

from phi.regime.models.clustering import ClusteringRegimeDetector
from phi.regime.models.deep import DeepRegimeDetector
from phi.regime.models.hmm import HMMRegimeDetector

__all__ = ["HMMRegimeDetector", "ClusteringRegimeDetector", "DeepRegimeDetector"]
