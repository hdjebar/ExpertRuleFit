"""Serializable rule specifications for ExpertRuleFit.

Replaces lambda-based rule evaluate functions with a declarative DSL that
can be serialized to JSON/dict, enabling:
    - ``joblib.dump`` / ``pickle`` without lambda issues
    - Audit bundles (JSON export of full rule set)
    - SQL code generation
    - Model governance (versioned rule definitions)

Usage:
    from expertrulefit.rule_spec import RuleSpec

    spec = RuleSpec(
        name="CSSF: High cash ratio",
        clauses=[{"feature": "cash_ratio", "op": ">", "threshold": 0.4}],
    )

    # Compile to a callable
    fn = spec.compile(feature_names)
    result = fn(X, feature_names)  # ndarray of {0.0, 1.0}

    # Serialize
    d = spec.to_dict()
    spec2 = RuleSpec.from_dict(d)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


_VALID_OPS = {"<", "<=", ">", ">=", "==", "!="}


@dataclass
class RuleSpec:
    """Declarative, serializable rule definition.

    A rule is a conjunction (AND) of one or more clauses. Each clause
    compares a named feature against a threshold with a comparison operator.

    Parameters
    ----------
    name : str
        Human-readable rule name (e.g., "CSSF: High cash ratio").
    clauses : list of dict
        Each dict has keys ``feature`` (str), ``op`` (str), ``threshold`` (float).
        Valid operators: ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``.
    invert : bool, default=False
        If True, the rule output is ``1 - base``, i.e., the rule fires when
        the conjunction is NOT met.
    metadata : dict, optional
        Arbitrary metadata for governance (e.g., ``source``, ``policy_id``,
        ``jurisdiction``, ``owner``, ``rationale``, ``created_at``).
    """

    name: str
    clauses: List[Dict[str, Any]]
    invert: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.clauses:
            raise ValueError("RuleSpec must have at least one clause")
        for i, c in enumerate(self.clauses):
            for key in ("feature", "op", "threshold"):
                if key not in c:
                    raise ValueError(
                        f"Clause {i} missing required key '{key}': {c}"
                    )
            if c["op"] not in _VALID_OPS:
                raise ValueError(
                    f"Clause {i} has invalid op '{c['op']}', "
                    f"must be one of {_VALID_OPS}"
                )

    def compile(self, feature_names: list[str]) -> Callable:
        """Compile this spec into a callable compatible with ExpertRuleFit.

        Parameters
        ----------
        feature_names : list of str
            Feature names matching columns of X.

        Returns
        -------
        evaluate : callable(X, feature_names) -> ndarray of float64
        """
        fn_to_idx = {fn: i for i, fn in enumerate(feature_names)}

        # Pre-validate feature references
        for c in self.clauses:
            if c["feature"] not in fn_to_idx:
                raise KeyError(
                    f"Rule '{self.name}': feature '{c['feature']}' not found "
                    f"in feature_names"
                )

        # Build index-based clause list for fast evaluation
        compiled_clauses = [
            (fn_to_idx[c["feature"]], c["op"], float(c["threshold"]))
            for c in self.clauses
        ]
        invert = self.invert

        _op_map = {
            "<": np.less,
            "<=": np.less_equal,
            ">": np.greater,
            ">=": np.greater_equal,
            "==": np.equal,
            "!=": np.not_equal,
        }

        def evaluate(X, feature_names):
            X = np.asarray(X, dtype=np.float64)
            result = np.ones(X.shape[0], dtype=bool)
            for col_idx, op, threshold in compiled_clauses:
                result &= _op_map[op](X[:, col_idx], threshold)
            out = result.astype(np.float64)
            return (1.0 - out) if invert else out

        return evaluate

    def to_rule_dict(
        self, feature_names: Optional[list[str]] = None,
        category: str = "confirmatory",
    ) -> dict:
        """Convert to a rule dict ready for ExpertRuleFit.fit().

        Parameters
        ----------
        feature_names : list of str, optional
            Required to compile the evaluate function.  If None, the
            returned dict will NOT have an ``evaluate`` key and must
            be compiled later.
        category : str, default="confirmatory"
            Either "confirmatory" or "optional".

        Returns
        -------
        rule_dict : dict
            Has keys ``name``, ``category``, and optionally ``evaluate``.
        """
        d = {"name": self.name, "category": category}
        if feature_names is not None:
            d["evaluate"] = self.compile(feature_names)
        return d

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "name": self.name,
            "clauses": [dict(c) for c in self.clauses],
            "invert": self.invert,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuleSpec":
        """Deserialize from a plain dict."""
        return cls(
            name=d["name"],
            clauses=d["clauses"],
            invert=d.get("invert", False),
            metadata=d.get("metadata", {}),
        )

    def to_sql(self, *, alias: Optional[str] = None) -> str:
        """Generate a SQL CASE WHEN expression for this rule.

        Parameters
        ----------
        alias : str, optional
            Column alias for the result. Defaults to a sanitized rule name.

        Returns
        -------
        sql : str
        """
        conditions = []
        for c in self.clauses:
            feat = c["feature"]
            op = c["op"]
            threshold = c["threshold"]
            conditions.append(f'"{feat}" {op} {threshold}')

        conjunction = " AND ".join(conditions)

        if self.invert:
            where = f"NOT ({conjunction})"
        else:
            where = conjunction

        if alias is None:
            alias = self.name.replace(" ", "_").replace(":", "")

        return f'CASE WHEN {where} THEN 1 ELSE 0 END AS "{alias}"'

    def __repr__(self) -> str:
        parts = []
        for c in self.clauses:
            parts.append(f"{c['feature']} {c['op']} {c['threshold']}")
        expr = " AND ".join(parts)
        if self.invert:
            expr = f"NOT ({expr})"
        return f"RuleSpec('{self.name}': {expr})"


def save_rule_specs(specs: list[RuleSpec], path: str | Path) -> None:
    """Save a list of RuleSpec to a JSON file."""
    data = [s.to_dict() for s in specs]
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def load_rule_specs(path: str | Path) -> list[RuleSpec]:
    """Load a list of RuleSpec from a JSON file."""
    data = json.loads(Path(path).read_text())
    return [RuleSpec.from_dict(d) for d in data]
