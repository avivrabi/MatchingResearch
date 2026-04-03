from config import NUM_PARTICIPANTS, FIXED_K, VARYING_RATIO_MAX_K
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

    def run_dynamic_matching(self, k, method="greedy"):
        """
        Iterates year by year over the follow-up period.
        At each year, participants reaching their surgery_age become treated and are matched with k controls.
        If a matched control later gets surgery, they are censored in their old set and become newly treated.

        Args:
            k: number of controls per treated participant
            method: "greedy" for varying-ratio, "optimal" for fixed-k

        Returns:
            all_matches: list of match dicts with "match_time"
            transitioned_controls: dict {participant_id: surgery_age}
        """
        # Everyone starts in the unmatched pool
        unmatched_pool = set(self.all_participants)

        all_matches = []
        transitioned_controls = {}
        all_skipped = []  # treated participants that could not be matched

        surgery_ages = sorted(set(p.surgery_age for p in self.all_participants if p.surgery_age is not None))
        if not surgery_ages:
            print("No participants with surgery found.")
            return all_matches, transitioned_controls

        for current_age in surgery_ages:
            # 1. Find unmatched participants getting surgery this year
            newly_treated_from_pool = [p for p in unmatched_pool if p.surgery_age == current_age and p.observed_age >= current_age]

            # 2. Find matched controls transitioning (getting surgery this year)
            transitioning_controls = []
            for match_idx, match in enumerate(all_matches):
                for control in match["control"]:
                    if control.surgery_age == current_age and control.observed_age >= current_age:
                        transitioning_controls.append(control)
                        transitioned_controls[control.id] = current_age

            # Combine all newly treated participants
            all_newly_treated = newly_treated_from_pool + transitioning_controls

            # Remove newly treated from unmatched pool to make sure every participant is only matched once
            for p in newly_treated_from_pool:
                unmatched_pool.discard(p)

            # In this case no matching is needed
            if not all_newly_treated:
                continue

            # Available controls: still in the pool, alive at current_age, no surgery yet or surgery in future
            available = [p for p in unmatched_pool
                         if p.observed_age >= current_age
                         and (p.surgery_age is None or p.surgery_age > current_age)]

            if not available:
                print(f"  Warning: {len(all_newly_treated)} treated participants at age {current_age} could not be matched (no available controls).")
                all_skipped.extend([(p, current_age, "no available controls") for p in all_newly_treated])
                continue

            # 3. Match newly treated with controls
            if method == "optimal":
                matcher = Matcher(all_newly_treated, list(available), fixed_k=k)
                try:
                    batch_matches = matcher.fixed_k_matching()
                # TODO: do we want this fallback or just add them to skipped?
                except ValueError:
                    print("Not enough controls for optimal — fall back to greedy for this batch")
                    matcher = Matcher(all_newly_treated, list(available), varying_ratio_max_k=k)
                    batch_matches = matcher.varying_ratio_matching()
            else: # method == "greedy"
                matcher = Matcher(all_newly_treated, list(available), varying_ratio_max_k=k)
                batch_matches = matcher.varying_ratio_matching()

            all_skipped.extend([(p, current_age, "no match within caliper") for p in matcher.skipped_participants])

            # Removes matched controls from the unmatched_pool so they can't be matched again
            for match in batch_matches:
                for c in match["control"]:
                    unmatched_pool.discard(c)
                match["match_time"] = current_age # Year of matching to track duration in analysis
                all_matches.append(match)

        # Log skipped participants to file
        if all_skipped:
            log_path = f"skipped_participants_{method}.txt"
            with open(log_path, "w") as f:
                f.write(f"Skipped participants for method: {method}\n")
                f.write(f"Total skipped: {len(all_skipped)}\n\n")
                for participant, age, reason in all_skipped:
                    f.write(f"Participant {participant.id} at age {age} "
                            f"(propensity={participant.propensity_score:.4f}) "
                            f"- reason: {reason}\n")
            print(f"  {len(all_skipped)} skipped participants logged to {log_path}")

        n_transitions = len(transitioned_controls)
        n_skipped = len(all_skipped)
        print(f"Dynamic matching complete: {len(all_matches)} matched sets, {n_transitions} control-to-treated transitions, {n_skipped} skipped.")
        return all_matches, transitioned_controls


    def run_analysis(self):
        """
        Runs dynamic matching + Stratified Cox Proportional Hazard analysis for both methods.
        HERE WE DETERMINE THE EXPERIMENTS MADE.
        """
        print("\n--- Fixed-K Dynamic Matching (Optimal) ---")
        fixed_k_matches, fixed_k_censored = self.run_dynamic_matching(k=FIXED_K, method="optimal")

        print("\n--- Varying-Ratio Dynamic Matching (Greedy) ---")
        varying_ratio_matches, varying_ratio_censored = self.run_dynamic_matching(k=VARYING_RATIO_MAX_K, method="greedy")

        fixed_k_analyzer = Analyzer(fixed_k_matches,
                                    k=FIXED_K,
                                    transitioned_controls=fixed_k_censored,
                                    method_name="Fixed-K (Dynamic)")
        fixed_k_analyzer.print_summary()

        varying_ratio_analyzer = Analyzer(varying_ratio_matches,
                                          k=VARYING_RATIO_MAX_K,
                                          transitioned_controls=varying_ratio_censored,
                                          method_name="Varying-Ratio (Dynamic)")
        varying_ratio_analyzer.print_summary()

        return fixed_k_analyzer, varying_ratio_analyzer


if __name__ == "__main__":
    experiment = Experiment()
    print(f"Running experiment with {experiment.num_participants} participants.")
    experiment.run_analysis()
