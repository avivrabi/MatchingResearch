import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


class Analyzer:
    """
    Analyzes matched cohorts using Stratified Cox Proportional Hazards.
    Each matched set (1 treated + k controls) forms a stratum.
    """
    def __init__(self, matches: list, k: int = 0, transitioned_controls: dict = None, method_name: str = ""):
        """
        Args:
            matches: list of dicts with {"treated": Participant, "control": [Participant, ...], "match_time": int}
            k: number of controls per treated participant used in matching
            transitioned_controls: dict {participant_id: surgery_age} for controls that transitioned to treated (censored at surgery_age)
            method_name: label for the matching method
        """
        self.matches = matches
        self.k = k
        self.transitioned_controls = transitioned_controls or {}
        self.method_name = method_name
        self.df = self.build_dataframe() # dataframe for Stratified Cox
        self.cox_model = None            # cox model that will be fit and used for analysis

    def build_dataframe(self) -> pd.DataFrame:
        """
        Converts matched sets into a DataFrame suitable for Stratified Cox PH.
        Schema: matched_set_id (int), duration (int), event_observed (0/1), treatment (0/1).
        Handles dynamic censoring: controls that transitioned to treated are censored at surgery_age.
        """
        rows = []
        for set_id, match in enumerate(self.matches):
            match_time = match.get("match_time", 0)
            treated = match["treated"]
            rows.append(self.participant_to_row(treated, set_id, match_time, is_treatment=True))

            for control in match["control"]:
                censor_age = self.transitioned_controls.get(control.id, None)
                rows.append(self.participant_to_row(control, set_id, match_time, is_treatment=False, censor_age=censor_age))

        return pd.DataFrame(rows)

    @staticmethod
    def participant_to_row(p, set_id: int, match_time: int, is_treatment: bool, censor_age: int = None) -> dict:
        """
        Converts a Participant into a row for the survival DataFrame.
        - duration: time from match_time to the observed event/censoring.
        - event_observed: 1 if cancer occurred, 0 if censored.
        - If censor_age is set (control transitioned to treated), they are censored at that age.
        """
        if censor_age is not None and censor_age < p.observed_age:
            # Control transitioned to treated: censored at surgery_age
            duration = censor_age - match_time
            event_observed = 0  # Old cohort can't know even if they got cancer
        else:
            duration = p.observed_age - match_time
            # cancer_age == 120 is a sentinel meaning "no cancer in a lifetime"
            event_observed = 1 if (p.cancer_age == p.observed_age and p.cancer_age < 120) else 0

        return {
            "matched_set_id": set_id,
            "duration": duration,
            "event_observed": event_observed,
            "treatment": int(is_treatment),
        }

    def run_stratified_cox(self) -> CoxPHFitter:
        """
        Fits a Stratified Cox Proportional Hazard model.
        Strata = matched sets (each with its own baseline hazard).
        The model estimates a single treatment coefficient (log hazard ratio).
        """
        self.cox_model = CoxPHFitter()
        self.cox_model.fit(
            self.df,
            duration_col="duration",
            event_col="event_observed",
            strata=["matched_set_id"],
        )
        return self.cox_model

    def print_summary(self):
        """Prints a readable summary of all HR estimates."""
        if self.cox_model is None:
            self.run_stratified_cox()

        # Stratified Cox (already fitted)
        strat_s = self.cox_model.summary
        strat_hr = strat_s.loc["treatment", "exp(coef)"]
        strat_ci = f"[{strat_s.loc['treatment', 'exp(coef) lower 95%']:.4f}, {strat_s.loc['treatment', 'exp(coef) upper 95%']:.4f}]"
        strat_se = strat_s.loc["treatment", "se(coef)"]
        strat_p = strat_s.loc["treatment", "p"]

        # Unstratified Cox (pooled, no strata — equivalent to KM-based inference)
        unstrat_model = CoxPHFitter()
        unstrat_model.fit(
            self.df[["duration", "event_observed", "treatment"]],
            duration_col="duration", event_col="event_observed",
        )
        unstrat_s = unstrat_model.summary
        unstrat_hr = unstrat_s.loc["treatment", "exp(coef)"]
        unstrat_ci = f"[{unstrat_s.loc['treatment', 'exp(coef) lower 95%']:.4f}, {unstrat_s.loc['treatment', 'exp(coef) upper 95%']:.4f}]"
        unstrat_se = unstrat_s.loc["treatment", "se(coef)"]
        unstrat_p = unstrat_s.loc["treatment", "p"]

        # DGP ground truth: beta_s_age=-2.5 in sample_time_of_cancer
        dgp_beta = -2.5
        dgp_hr = np.exp(dgp_beta)

        n_strata = self.df["matched_set_id"].nunique()
        n_treated = int(self.df["treatment"].sum())
        n_controls = len(self.df) - n_treated
        n_events = int(self.df["event_observed"].sum())
        n_transitions = len(self.transitioned_controls)

        w = 78
        print(f"\n{'=' * w}")
        print(f"  {self.method_name}")
        print(f"{'=' * w}")
        print(f"  Matched sets: {n_strata}   Treated: {n_treated}   Controls: {n_controls}")
        print(f"  Events: {n_events}   Transitions: {n_transitions}")
        print(f"{'-' * w}")
        print(f"  {'Model':<18}{'HR':>10}{'95% CI':>24}{'SE':>10}{'p-value':>14}{'|Diff|':>10}")
        print(f"  {'-' * (w - 4)}")
        print(f"  {'DGP (truth)':<18}{dgp_hr:>10.4f}{'':>24}{'':>10}{'':>14}{'':>10}")
        print(f"  {'Stratified':<18}{strat_hr:>10.4f}{strat_ci:>24}{strat_se:>10.4f}{strat_p:>14.2e}{abs(strat_hr - dgp_hr):>10.4f}")
        print(f"  {'Unstratified':<18}{unstrat_hr:>10.4f}{unstrat_ci:>24}{unstrat_se:>10.4f}{unstrat_p:>14.2e}{abs(unstrat_hr - dgp_hr):>10.4f}")
        print(f"{'-' * w}")
        print(f"  => Treatment reduces cancer hazard by {(1 - strat_hr) * 100:.1f}% (Stratified Cox)")
        print(f"{'=' * w}\n")

    def get_variance_metrics(self) -> dict:
        """
        Computes variance-related metrics for the results.

        Returns:
            dict with variance metrics including
            - coef_se: standard error of the coefficient
            - hr_se: standard error of the hazard ratio
            - ci_width: width of 95% CI for the hazard ratio
        """
        if self.cox_model is None:
            self.run_stratified_cox()

        summary = self.cox_model.summary
        coef = summary.loc["treatment", "coef"]
        coef_se = summary.loc["treatment", "se(coef)"]
        hr = summary.loc["treatment", "exp(coef)"]
        hr_lower = summary.loc["treatment", "exp(coef) lower 95%"]
        hr_upper = summary.loc["treatment", "exp(coef) upper 95%"]
        ci_width = hr_upper - hr_lower

        return {
            "method": self.method_name,
            "coefficient": coef,
            "coef_se": coef_se,
            "hazard_ratio": hr,
            "hr_lower_95": hr_lower,
            "hr_upper_95": hr_upper,
            "ci_width": ci_width,
            "n_matched_sets": self.df["matched_set_id"].nunique(),
            "n_treated": int(self.df["treatment"].sum()),
            "n_controls": int(len(self.df) - self.df["treatment"].sum()),
        }
