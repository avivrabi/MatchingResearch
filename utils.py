"""
This file contains utility and helper functions.
"""

import numpy as np
from scipy.special import logit
from config import CLIP_EPS


def calculate_propensity_distance(control_propensity: float, treatment_propensity: float, is_logit: bool = True):
    if is_logit:
        return abs(logit(np.clip(control_propensity, CLIP_EPS, 1 - CLIP_EPS)) - logit(np.clip(treatment_propensity, CLIP_EPS, 1 - CLIP_EPS)))
    else:
        return abs(control_propensity - treatment_propensity)
