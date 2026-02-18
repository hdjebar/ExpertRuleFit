"""ExpertRuleFit — Reproducible interpretable ML for regulated environments."""

from .expert_rulefit import ExpertRuleFit
from .ebm_bridge import discover_interaction_rules
from .dual_model import DualModel

__version__ = "0.5.0"
__all__ = ["ExpertRuleFit", "discover_interaction_rules", "DualModel"]
