import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from config import VARYING_RATIO_MAX_K, FIXED_K, CALIPER, IS_LOGIT, INF
from config import CLIP_EPS


def calculate_propensity_distance(control_propensity: float, treatment_propensity: float, is_logit: bool = True):
    if is_logit:
        return abs(logit(np.clip(control_propensity, CLIP_EPS, 1 - CLIP_EPS)) - logit(np.clip(treatment_propensity, CLIP_EPS, 1 - CLIP_EPS)))
    else:
        return abs(control_propensity - treatment_propensity)

class Matcher:
    """
    This class is used to match a control participant to a treatment participant.
    It can use one of two paradigms:
    1. Fixed 1:k matching: each control participant is matched to k treatment participants.
       Matching is optimal, minimizing the sum of distances between control and treatment participants.
    2. Varying ratio matching: each control participant is matched to a varying number of treatment participants.
       Matching is greedy, matching each control participant to its closest treatment participant.
    """

    def __init__(self, treatment_participants, control_participants,
                 fixed_k: int = FIXED_K, varying_ratio_max_k: int = VARYING_RATIO_MAX_K,
                 caliper: float = CALIPER, is_logit: bool = IS_LOGIT):
        self.treatment_participants = treatment_participants
        self.control_participants = control_participants
        self.fixed_k = fixed_k
        self.varying_ratio_max_k = varying_ratio_max_k
        self.caliper = caliper
        self.is_logit = is_logit
        self.skipped_participants = [] # Participants that could not be matched

    def greedy_match_control_to_treated(self, treated_participant):
        """
        Returns the closest control participant to a given treated participant.
        Removes the matched control from self.control_participants so it is no longer available.
        Returns None if no control is found within the caliper.
        """
        min_distance = float('inf')
        greedy_match = None
        for control_participant in self.control_participants:
            distance = calculate_propensity_distance(treated_participant.propensity_score, control_participant.propensity_score, self.is_logit)
            if distance < min_distance and distance <= self.caliper:
                min_distance = distance
                greedy_match = control_participant
        if greedy_match is not None:
            self.control_participants.remove(greedy_match)
        return greedy_match

    def greedy_match_treated_to_control(self, control_participant, available_treated=None):
        """
        Returns the closest treated participant to a given control participant.
        Searches within available_treated (defaults to all treatment participants if not provided).
        Removes the control participant from self.control_participants once consumed.
        """
        if available_treated is None:
            available_treated = self.treatment_participants
        min_distance = float('inf')
        greedy_match = None
        for treated_participant in available_treated:
            distance = calculate_propensity_distance(treated_participant.propensity_score, control_participant.propensity_score, self.is_logit)
            if distance < min_distance and distance <= self.caliper:
                min_distance = distance
                greedy_match = treated_participant
        if greedy_match is not None:
            self.control_participants.remove(control_participant)
        return greedy_match

    def varying_ratio_matching(self):
        """
        Performs varying-ratio matching (two-phase greedy).
        Phase 1: Each treated patient is guaranteed at least one control (skipped if none within caliper).
        Phase 2: Remaining controls are greedily assigned up to varying_ratio_max_k per treated.

        Returns a list of dicts: {"treated": Participant, "control": [Participant, ...]}.
        Populates self.skipped_participants with treated participants that could not be matched.
        """
        matches = []
        self.skipped_participants = []

        # Phase 1: guarantee each treated patient gets at least one control.
        for treated_participant in self.treatment_participants:
            greedy_match = self.greedy_match_control_to_treated(treated_participant)
            if greedy_match is not None:
                matches.append({"treated": treated_participant, "control": [greedy_match]})
            else:
                self.skipped_participants.append(treated_participant)

        # Phase 2: greedily assign remaining controls, up to varying_ratio_max_k per treated patient.
        for control_participant in list(self.control_participants):
            available_treated = [m["treated"] for m in matches if len(m["control"]) < self.varying_ratio_max_k]
            best_treated = self.greedy_match_treated_to_control(control_participant, available_treated)
            if best_treated:
                for match in matches:
                    if match["treated"] == best_treated:
                        match["control"].append(control_participant)
                        break

        return matches

    def fixed_k_matching(self):
        """
        Performs optimal fixed 1:k matching using the Hungarian algorithm (scipy linear_sum_assignment).
        Each treated participant is assigned exactly fixed_k controls, minimizing the total sum of
        propensity score distances across all matches.

        Returns a list of dicts: {"treated": Participant, "control": [Participant, ...]}.
        """
        # First, make sure there are enough control participants.
        if len(self.control_participants) < self.fixed_k * len(self.treatment_participants):
            raise ValueError("Not enough control participants for fixed-k matching. Choose a smaller fixed-k value.")

        # Step 1: Build cost matrix of shape (n_treated * fixed_k) x n_controls.
        # Extract propensity scores into arrays and compute all pairwise distances in one
        # vectorized numpy operation (broadcasting) instead of a Python loop per cell.
        # Each treated row is then repeated fixed_k times to force the algorithm to pick
        # fixed_k distinct controls per treated participant.
        # Pairs exceeding the caliper are masked to INF so the algorithm avoids them.
        treated_scores = np.array([t.propensity_score for t in self.treatment_participants])
        control_scores = np.array([c.propensity_score for c in self.control_participants])

        if self.is_logit:
            treated_scores = logit(np.clip(treated_scores, 1e-6, 1 - 1e-6))
            control_scores = logit(np.clip(control_scores, 1e-6, 1 - 1e-6))

        # distance_matrix[i, j] = |treated_scores[i] - control_scores[j]|
        distance_matrix = np.abs(treated_scores[:, np.newaxis] - control_scores[np.newaxis, :])
        distance_matrix[distance_matrix > self.caliper] = INF

        cost_matrix = np.repeat(distance_matrix, self.fixed_k, axis=0)  # shape: (n_treated * fixed_k, n_controls)

        # Step 2: Run the Hungarian algorithm to get the optimal assignment.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Step 3: Build matches. Any assigned pair still carrying INF means
        # the algorithm had no valid control within the caliper for that treated participant.
        matches = [{"treated": t, "control": []} for t in self.treatment_participants]
        for row, col in zip(row_ind, col_ind):
            treated_idx = row // self.fixed_k
            treated = self.treatment_participants[treated_idx]
            control = self.control_participants[col]
            distance = cost_matrix[row, col]
            if distance >= INF:
                raise ValueError(
                    f"Could not find {self.fixed_k} controls within caliper for treated "
                    f"participant {treated.id}. Consider widening the caliper or adding more controls."
                )
            matches[treated_idx]["control"].append(control)

        return matches
