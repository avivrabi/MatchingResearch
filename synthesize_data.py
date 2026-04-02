import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Set seed for reproducibility
GLOBAL_SEED = 42
rng = np.random.default_rng(GLOBAL_SEED)

# Israel-specific mortality constants (Gompertz-Makeham)
ISR_ALPHA = 0.0001  # Baseline biological vulnerability
ISR_BETA = 0.075    # Rate of aging (approx. 9-year doubling time)
ISR_GAMMA = 0.0005  # Background non-age-related risk


### 1st phase sampling functions ###
def sample_discovery_age():
    """D ~ uni[25, 40] discrete."""
    return np.random.randint(25, 41)

def sample_family_risk():
    # TODO: Changed from Bernoulli(0.5) to Uniform(0,1) because the binary variable
    #  caused a bimodal propensity score distribution (two distinct clusters).
    """F ~ Uniform(0, 1). Continuous family risk score."""
    return np.random.uniform(0, 1)

def sample_censor_age():
    """C ~ uni[25, 150] discrete."""
    # TODO: what is the censor rate? needs to be ~20% if too high increase 151 to ~200
    return np.random.randint(25, 151)

def sample_other_explanations():
    """X ~ uni[-1, 1] continuous. 1 is bad (higher risk), -1 is good."""
    return np.random.uniform(-1, 1)


### 2nd phase sampling functions ###
def sample_surgery_at_age(d, p=0.02):
    """
    TODO: p can be tuned
    S: Age of surgery.
    Logic: Surgery happens at age D + k with a probability p*(1-p)^k.
    Constraint: S > D and S < 50. If no surgery by 50, the patient age of surgery is 'None'.
    """
    # k follows a geometric distribution (number of failures before success)
    k = np.random.geometric(p) - 1
    s = d + k
    if s < 50:
        return s
    return None # No surgery performed before age limit

def sample_time_of_cancer(age_start, f, x, s_age, beta_f=3, beta_x=1, beta_s_age=-2.5):
    """
    T1: Age of cancer onset.
    Logic: P(age) = baseline_risk(age) * exp(beta_f*F + beta_x*X + beta_s*Indicator)
    """
    for age in range(age_start, 121):
        # Baseline risk increases linearly as a simple approximation for P0,1(t)
        baseline_hazard = 0.003 * (age / 25)

        # Check if surgery has already happened by this age
        has_had_surgery = 1 if (s_age is not None and age >= s_age) else 0

        # Calculate risk multiplier
        exponent = (beta_f * f) + (beta_x * x) + (beta_s_age * has_had_surgery)
        hazard = baseline_hazard * np.exp(exponent)

        if np.random.random() < hazard:
            return age

    return 120 # Did not get cancer in a lifetime

def sample_other_factor_death(age_start, x, beta_x=0.5):
    """
    T2: Age of death (non-cancer).
    Logic: Uses an Israel-calibrated Gompertz-Makeham curve.
    λ(a) = (α*e^(β*a) + γ) * exp(beta_x * X)
    """
    for age in range(age_start, 121):
        # Gompertz-Makeham baseline for Israel
        baseline_hazard = ISR_ALPHA * np.exp(ISR_BETA * age) + ISR_GAMMA

        # Apply individual covariate X
        hazard = baseline_hazard * np.exp(beta_x * x)

        if np.random.random() < hazard:
            return age

    return 120


### Synthetic data generation function ###
def generate_patient_data(n=1000):
    patients = []
    for i in range(n):
        d = sample_discovery_age()
        f = sample_family_risk()
        x = sample_other_explanations()
        c = sample_censor_age()
        s = sample_surgery_at_age(d)

        # Survival events
        t1 = sample_time_of_cancer(d, f, x, s)
        t2 = sample_other_factor_death(d, x)

        # The observed 'End of Study' age is the minimum of all stopping events
        event_age = min(t1, t2, c)

        patients.append({
            "id": i,
            "discovery_age": d,
            "family_risk": f,
            "other_covariates": x,
            "censoring_age": c,
            "surgery_age": s,
            "cancer_age": t1,
            "death_age": t2,
            "observed_age": event_age
        })

    # Calculate propensity scores for each patient using logistic regression based on its covariates
    # Propensity score = P(cancer|covariates)
    X = np.array([[p["discovery_age"], p["family_risk"], p["other_covariates"]] for p in patients])
    y = np.array([1 if p["cancer_age"] < 120 else 0 for p in patients])

    model = LogisticRegression()
    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    for p, score in zip(patients, probabilities):
        p["propensity_score"] = score

    return patients

### Data Analysis ###
def run_eda(patients_list):
    df = pd.DataFrame(patients_list)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'EDA: Synthetic Israeli Patient Population (N={len(patients_list)})', fontsize=16)

    # 1. Mortality Curve (Includes deaths up to 120)
    sns.histplot(df['death_age'], bins=30, kde=True, ax=axes[0, 0], color='gray')
    axes[0, 0].set_title('Death Age Distribution (T2)')
    axes[0, 0].axvline(83, color='red', linestyle='--', label='Avg ISR Life Expectancy (~83)')
    axes[0, 0].legend()

    # 2. Cancer Onset by Family Risk (Use 120 as the "No Cancer" marker)
    cancer_patients = df[df['cancer_age'] < 120].copy()
    cancer_patients['family_risk_bin'] = pd.cut(cancer_patients['family_risk'], bins=5)
    sns.boxplot(x='family_risk_bin', y='cancer_age', data=cancer_patients, ax=axes[0, 1])
    axes[0, 1].set_title('Cancer Onset Age (T1) by Family Risk')
    axes[0, 1].set_xlabel('Family Risk (binned)')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # 3. Surgery Age
    surgery_patients = df[df['surgery_age'].notnull()]
    sns.histplot(surgery_patients['surgery_age'], bins=15, kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Surgery Age (S) Distribution')
    axes[1, 0].set_xlim(25, 50)

    # 4. Cancer Rate vs Covariate X
    df['got_cancer'] = df['cancer_age'] < 120
    sns.pointplot(x=pd.cut(df['other_covariates'], bins=5), y='got_cancer', data=df, ax=axes[1, 1], hue=pd.cut(df['other_covariates'], bins=5), legend=False)
    axes[1, 1].set_title('Cancer Probability vs. Other Factors (X)')

    # 5. Propensity Score Distribution (Treated vs Control)
    treated = df[df['surgery_age'].notnull()]
    control = df[df['surgery_age'].isnull()]
    mean_score = df['propensity_score'].mean()
    sns.histplot(treated['propensity_score'], kde=True, stat="density", label="Treated", color="salmon", alpha=0.5, ax=axes[2, 0])
    sns.histplot(control['propensity_score'], kde=True, stat="density", label="Control", color="steelblue", alpha=0.5, ax=axes[2, 0])
    axes[2, 0].axvline(mean_score, color="black", linestyle="--", linewidth=1.5, label=f"Mean = {mean_score:.4f}")
    axes[2, 0].set_xlabel("Propensity Score")
    axes[2, 0].set_ylabel("Density")
    axes[2, 0].set_title("Propensity Score Distribution")
    axes[2, 0].legend()

    # 6. Average Propensity Score per Discovery Age
    avg_by_age = df.groupby('discovery_age')['propensity_score'].mean()
    axes[2, 1].plot(avg_by_age.index, avg_by_age.values, marker="o", color="teal")
    axes[2, 1].set_xlabel("Discovery Age")
    axes[2, 1].set_ylabel("Average Propensity Score")
    axes[2, 1].set_title("Average Propensity Score per Discovery Age")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print(f"Average Death Age: {df['death_age'].mean():.2f}")
    print(f"Cancer Prevalence: {df['got_cancer'].mean() * 100:.1f}%")


if __name__ == "__main__":
    data = generate_patient_data(10000)
    run_eda(data)
