from config import NUM_PARTICIPANTS, FIXED_K, VARYING_RATIO_MAX_K, FOLLOW_UP_YEARS
from synthesize_data import generate_patient_data
from participant import Participant
from matching import Matcher
from analysis import Analyzer


class Experiment:
    """
    This class simulates a dynamic matching experiment.
    At each year, participants who undergo surgery become treated and are matched with controls.
    If a matched control later gets surgery, they are censored in their old set and re-matched as treated.
    """
    def __init__(self, num_participants: int = NUM_PARTICIPANTS):
        self.num_participants = num_participants

        # Generate raw data and wrap each record into a Participant object
        raw_data = generate_patient_data(num_participants)
        print(f"Generated {len(raw_data)} records.")
        self.all_participants = [Participant(p) for p in raw_data]

        n_with_surgery = len([p for p in self.all_participants if p.surgery_age is not None])
        print(f"Participants with surgery: {n_with_surgery}, without: {len(self.all_participants) - n_with_surgery}")

    def run_dynamic_matching(self, k, method="greedy", follow_up_years=FOLLOW_UP_YEARS):
        """
        Iterates year by year over the follow-up period.
        At each year, participants reaching their surgery_age become treated and are matched with k controls.
        If a matched control later gets surgery, they are censored in their old set and become newly treated.

        Args:
            k: number of controls per treated participant
            method: "greedy" for varying-ratio, "optimal" for fixed-k (Hungarian algorithm)
            follow_up_years: number of years to simulate

        Returns:
            all_matches: list of match dicts with "match_time"
            censored_controls: dict {participant_id: surgery_age}
        """
        # Everyone starts in the unmatched pool
        unmatched_pool = set(self.all_participants)

        all_matches = []
        censored_controls = {}

        min_age = min(p.discovery_age for p in self.all_participants)
        max_age = min_age + follow_up_years

        for current_age in range(min_age, max_age + 1):
            # 1. Find unmatched participants getting surgery this year
            newly_treated_from_pool = [p for p in unmatched_pool
                                       if p.surgery_age == current_age
                                       and p.observed_age >= current_age]

            # 2. Find matched controls transitioning (getting surgery this year)
            transitioning_controls = []
            for match_idx, match in enumerate(all_matches):
                for control in match["control"]:
                    if (control.surgery_age == current_age
                            and control.id not in censored_controls
                            and control.observed_age >= current_age):
                        transitioning_controls.append(control)
                        censored_controls[control.id] = current_age

            # Combine all newly treated participants
            all_newly_treated = newly_treated_from_pool + transitioning_controls

            # Remove newly treated from unmatched pool
            for p in newly_treated_from_pool:
                unmatched_pool.discard(p)

            if not all_newly_treated:
                continue

            # Available controls: still in pool, alive at current_age, no surgery yet or surgery in future
            available = [p for p in unmatched_pool
                         if p.observed_age >= current_age
                         and (p.surgery_age is None or p.surgery_age > current_age)]

            if not available:
                continue

            # 3. Match newly treated with controls
            if method == "optimal" and len(all_newly_treated) > 0:
                # Optimal (Hungarian): batch all newly treated at this time step
                matcher = Matcher(all_newly_treated, list(available), fixed_k=k)
                try:
                    batch_matches = matcher.fixed_k_matching()
                except ValueError:
                    # Not enough controls for optimal — fall back to greedy for this batch
                    batch_matches = self._greedy_match_batch(all_newly_treated, available, k)
            else:
                # Greedy: match one treated at a time
                batch_matches = self._greedy_match_batch(all_newly_treated, available, k)

            for match in batch_matches:
                for c in match["control"]:
                    unmatched_pool.discard(c)
                match["match_time"] = current_age
                all_matches.append(match)

        n_transitions = len(censored_controls)
        print(f"Dynamic matching complete: {len(all_matches)} matched sets, {n_transitions} control-to-treated transitions.")
        return all_matches, censored_controls

    @staticmethod
    def _greedy_match_batch(treated_list, available_controls, max_k):
        """
        Two-phase varying-ratio greedy matching for a batch of newly treated participants.
        Phase 1: Guarantee each treated gets at least 1 control.
        Phase 2: Distribute remaining controls up to max_k per treated.
        """
        matcher = Matcher(treated_list, list(available_controls), varying_ratio_max_k=max_k)

        # Phase 1: guarantee each treated gets at least 1 control
        batch_matches = []
        for treated in treated_list:
            greedy_match = matcher.greedy_match_control_to_treated(treated)
            if greedy_match is not None:
                batch_matches.append({"treated": treated, "control": [greedy_match]})

        # Phase 2: distribute remaining controls up to max_k per treated
        for control in list(matcher.control_participants):
            available_treated = [m["treated"] for m in batch_matches if len(m["control"]) < max_k]
            best_treated = matcher.greedy_match_treated_to_control(control, available_treated)
            if best_treated:
                for match in batch_matches:
                    if match["treated"] == best_treated:
                        match["control"].append(control)
                        break

        return batch_matches

    def run_analysis(self):
        """Runs dynamic matching + Stratified Cox PH analysis for both methods."""
        print("\n--- Fixed-K Dynamic Matching (Optimal) ---")
        fixed_k_matches, fixed_k_censored = self.run_dynamic_matching(k=FIXED_K, method="optimal")

        print("\n--- Varying-Ratio Dynamic Matching (Greedy) ---")
        varying_ratio_matches, varying_ratio_censored = self.run_dynamic_matching(k=VARYING_RATIO_MAX_K, method="greedy")

        fixed_k_analyzer = Analyzer(fixed_k_matches, k=FIXED_K, censored_controls=fixed_k_censored, method_name="Fixed-K (Dynamic)")
        fixed_k_analyzer.print_summary()

        varying_ratio_analyzer = Analyzer(varying_ratio_matches, k=VARYING_RATIO_MAX_K, censored_controls=varying_ratio_censored, method_name="Varying-Ratio (Dynamic)")
        varying_ratio_analyzer.print_summary()

        return fixed_k_analyzer, varying_ratio_analyzer


if __name__ == "__main__":
    experiment = Experiment()
    print(f"Running experiment with {experiment.num_participants} participants.")
    experiment.run_analysis()
