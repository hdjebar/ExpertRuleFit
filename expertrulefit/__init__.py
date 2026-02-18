"""ExpertRuleFit — Reproducible interpretable ML for regulated environments."""

from .expert_rulefit import ExpertRuleFit
from .ebm_bridge import discover_interaction_rules

__version__ = "0.4.0"
__all__ = ["ExpertRuleFit", "discover_interaction_rules"]
