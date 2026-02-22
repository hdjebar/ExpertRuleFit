"""ExpertRuleFit — Reproducible interpretable ML for regulated environments."""

from .expert_rulefit import ExpertRuleFit
from .ebm_bridge import discover_interaction_rules
from .dual_model import DualModel
from .dual_glass_box import DualGlassBox, EBMRuleExtractor, ExtractedRule

# Single-source version: prefer installed package metadata (kept in sync with
# pyproject.toml via [tool.setuptools.dynamic] → _version.__version__), fall
# back to _version.py for editable / development installs.
try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("expertrulefit")
except Exception:
    from ._version import __version__

__all__ = [
    "ExpertRuleFit",
    "discover_interaction_rules",
    "DualModel",
    "DualGlassBox",
    "EBMRuleExtractor",
    "ExtractedRule",
]
