"""ExpertRuleFit — Reproducible interpretable ML for regulated environments."""

from .expert_rulefit import ExpertRuleFit
from .ebm_bridge import discover_interaction_rules
from .dual_model import DualModel
from .dual_glass_box import DualGlassBox, EBMRuleExtractor, ExtractedRule

__version__ = "0.6.0"
__all__ = [
    "ExpertRuleFit",
    "discover_interaction_rules",
    "DualModel",
    "DualGlassBox",
    "EBMRuleExtractor",
    "ExtractedRule",
]
