"""Causal prediction-scale calibration (policy TBD-1).

COR and unclipped APS are invariant to positive rescaling, but AR scales
inversely with prediction scale and the frozen alpha clip makes APS
scale-sensitive once it binds. The final calibration rule is a team policy that
will be frozen before hidden scoring; until then the contract default is
``identity`` (scores are treated as return-unit alphas as submitted).

Any future rule must be causal: fitted only within the anchor's training
history, never on the predictions being scored. The evaluator applies
``calibrate -> clip -> score``.
"""

from __future__ import annotations

import numpy as np

from . import contract


def calibrate(alpha: np.ndarray, rule: str | None = None) -> np.ndarray:
    rule = rule or contract.CALIBRATION_RULE
    if rule == "identity":
        return alpha
    raise ValueError(f"Unknown calibration rule {rule!r} (TBD-1 not yet frozen)")
