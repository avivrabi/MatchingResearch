class Participant:
    """
    Represents a participant in the trial.
    """
    def __init__(self, patient_info:dict):
        self.id = patient_info["id"]
        self.discovery_age = patient_info["discovery_age"]
        self.family_risk = patient_info["family_risk"]
        self.other_covariates = patient_info["other_covariates"]
        self.censoring_age = patient_info["censoring_age"]
        self.surgery_age = patient_info["surgery_age"]
        self.cancer_age = patient_info["cancer_age"]
        self.death_age = patient_info["death_age"]
        self.observed_age = patient_info["observed_age"]
        self.is_treatment = self.surgery_age is not None
        self.propensity_score = patient_info["propensity_score"]
