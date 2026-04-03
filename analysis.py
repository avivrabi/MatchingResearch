import pandas as pd
from lifelines import CoxPHFitter


class Analyzer:
    """
    Analyzes matched cohorts using Stratified Cox Proportional Hazards.
    Each matched set (1 treated + k controls) forms a stratum.
    """
    def __init__(self, matches: list, k: int = 0, censored_controls: dict = None, method_name: str = ""):
        """
        Args:
            matches: list of dicts with {"treated": Participant, "control": [Participant, ...], "match_time": int}
            k: number of controls per treated participant used in matching
            censored_controls: dict {participant_id: surgery_age} for controls that transitioned to treated (censored at surgery_age)
            method_name: label for the matching method
        """
        self.matches = matches
        self.k = k
        self.censored_controls = censored_controls or {}
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
                censor_age = self.censored_controls.get(control.id, None)
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
        """Prints a readable summary of the Stratified Cox PH results."""
        if self.cox_model is None:
            self.run_stratified_cox()

        summary = self.cox_model.summary
        coef = summary.loc["treatment", "coef"]
        hr = summary.loc["treatment", "exp(coef)"]
        hr_lower = summary.loc["treatment", "exp(coef) lower 95%"]
        hr_upper = summary.loc["treatment", "exp(coef) upper 95%"]
        p_value = summary.loc["treatment", "p"]
        n_strata = self.df["matched_set_id"].nunique()
        n_total = len(self.df)
        n_treated = self.df["treatment"].sum()
        n_controls = n_total - n_treated
        n_events = self.df["event_observed"].sum()
        n_censored_transitions = len(self.censored_controls)

        print(f"\n{'='*60}")
        print(f" Stratified Cox PH Results - {self.method_name}")
        print(f"{'='*60}")
        print(f"  k (controls per treated): {self.k}")
        print(f"  Matched sets (strata):  {n_strata}")
        print(f"  Total participants:     {n_total}")
        print(f"    Treated:              {n_treated}")
        print(f"    Controls:             {n_controls}")
        print(f"    Cancer events:        {n_events}")
        print(f"    Dynamic transitions:  {n_censored_transitions}")
        print(f"{'-'*60}")
        print(f"  Coefficient (beta):     {coef:.4f}")
        print(f"  Hazard Ratio (e^beta):  {hr:.4f}")
        print(f"    95% CI:               [{hr_lower:.4f}, {hr_upper:.4f}]")
        print(f"  p-value:                {p_value:.2e}")
        print(f"{'-'*60}")
        if hr < 1:
            reduction = (1 - hr) * 100
            print(f"  => Treatment reduces cancer hazard by {reduction:.1f}%")
        else:
            increase = (hr - 1) * 100
            print(f"  => Treatment increases cancer hazard by {increase:.1f}%")
        print(f"{'='*60}\n")
