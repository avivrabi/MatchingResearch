from config import NUM_PARTICIPANTS, FIXED_K, VARYING_RATIO_MAX_K
from synthesize_data import generate_patient_data
from participant import Participant
from matching import Matcher
from analysis import Analyzer


class Experiment:
    """
    This class simulates an experiment.
    """
    def __init__(self, num_participants: int = NUM_PARTICIPANTS):
        self.num_participants = num_participants

        # Generate raw data and wrap each record into a Participant object
        raw_data = generate_patient_data(num_participants)
        print(f"Generated {len(raw_data)} records.")
        self.all_participants = [Participant(p) for p in raw_data]
        print(f"Total treated participants: {len([p for p in self.all_participants if p.is_treatment])}")

        # Participants are split into treated (had surgery) and control (no surgery)
        self.treated_participants = [p for p in self.all_participants if p.is_treatment]
        self.control_participants = [p for p in self.all_participants if not p.is_treatment]

    def run_matching(self):
        """Runs both matching methods and prints the resulting matches."""
        # Each method needs its own Matcher with a fresh copy of the control list
        fixed_k_matcher = Matcher(self.treated_participants, list(self.control_participants), fixed_k=FIXED_K)
        varying_ratio_matcher = Matcher(self.treated_participants, list(self.control_participants), varying_ratio_max_k=VARYING_RATIO_MAX_K)

        varying_ratio_matches = varying_ratio_matcher.varying_ratio_matching()
        fixed_k_matches = fixed_k_matcher.fixed_k_matching()

        return fixed_k_matches, varying_ratio_matches

    def run_analysis(self):
        """Runs matching followed by Stratified Cox PH analysis on both methods."""
        fixed_k_matches, varying_ratio_matches = self.run_matching()

        fixed_k_analyzer = Analyzer(fixed_k_matches, method_name="Fixed-K")
        fixed_k_analyzer.print_summary()

        varying_ratio_analyzer = Analyzer(varying_ratio_matches, method_name="Varying-Ratio")
        varying_ratio_analyzer.print_summary()

        return fixed_k_analyzer, varying_ratio_analyzer


if __name__ == "__main__":
    experiment = Experiment()
    print(f"Running experiment with {experiment.num_participants} participants.")
    experiment.run_analysis()
