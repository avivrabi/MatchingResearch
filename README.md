# Dynamic 1:k Matching Analysis of Prophylactic Mastectomy Effectiveness

A research project that evaluates the long-term effectiveness of prophylactic mastectomy in high-risk breast cancer patients using **dynamic 1:k propensity score matching** with **Stratified Cox Proportional Hazards** analysis.

This project adapts the methodology of [Dagan et al. (2021)](https://www.nejm.org/doi/full/10.1056/NEJMoa2101765) from COVID-19 vaccination effectiveness to mastectomy effectiveness, extending it from 1:1 to 1:k matching with dynamic cohort transitions.

## Research Context

Breast cancer affects approximately 1 in 8 Israeli women. BRCA1/BRCA2 mutations, prevalent among Ashkenazi, Ethiopian, and Iraqi populations, account for 50-80% of hereditary cases. Prophylactic mastectomy is a risk-reduction intervention for high-risk individuals.

**The methodological gap this project addresses**: While methods exist for static cohorts with multiple controls and for dynamic 1:1 matching, the combination — *dynamic 1:k matching* — lacks established methodology. Key unresolved challenges include:
- Follow-up endpoint definition when matched controls receive treatment
- Appropriate estimators for dynamic cohorts with censored data
- Handling control-to-treated transitions without introducing bias

## How It Works

### The Pipeline (Step by Step)

```
synthesize_data.py → participant.py → experiment.py → matching.py → analysis.py → visualizations.py
```

#### 1. Data Generation (`synthesize_data.py`)
Generates synthetic Israeli patient data for N participants. Each patient has:
- **Discovery age** (D): Age of high-risk gene discovery, uniform [25, 40]
- **Family risk** (F): Continuous risk score, uniform [0, 1]
- **Other covariates** (X): Additional risk factors, uniform [-1, 1] where X=1 means the patient is high-risk.
- **Surgery age** (S): Prophylactic mastectomy age, geometric distribution from D with covariate-dependent probability p = σ(base + β_f·F + β_x·X). Higher family risk and risk factors increase surgery likelihood. (or None if no surgery before 50)
- **Cancer age** (T1): Sampled via hazard model with baseline risk, family risk, other factors, and surgery effect
- **Death age** (T2): Israel-calibrated Gompertz-Makeham mortality model
- **Observed age**: min(cancer, death, censoring) — the study endpoint
- **Propensity score**: P(treatment | covariates), estimated via logistic regression on all patient covariates

Also includes `run_eda()` for exploratory data analysis.

#### 2. Participant Wrapper (`participant.py`)
Simple data class wrapping each patient dict into an object with attributes like `surgery_age`, `propensity_score`, `is_treatment`, etc.

#### 3. Dynamic Matching (`experiment.py`)
The core innovation. The `Experiment` class orchestrates year-by-year dynamic matching:

1. **All participants start in an unmatched pool**
2. **For each surgery age** (iterating over all unique surgery ages in the data):
   - **Find newly treated**: participants in the unmatched pool whose surgery happens at this age
   - **Find transitioning controls**: already-matched controls whose surgery happens at this age → they are censored in their old matched set and become newly treated
   - **Combine** both groups as newly treated participants
   - **Find available controls**: participants still in the unmatched pool who are alive and haven't had surgery yet
   - **Match** newly treated with available controls using the chosen algorithm
   - **Remove** matched controls from the pool (each participant can be control only once)
3. **Log skipped participants** (those who couldn't be matched) to CSV with reasons

The experiment runs **4 configurations** to compare two matching paradigms:

**Fixed-K**: Every treated participant gets exactly k controls. Uses the Hungarian algorithm to find the globally optimal assignment that minimizes total propensity score distance across all pairs.

**Varying-Ratio**: Each treated participant gets between 1 and max_k controls, depending on availability. Uses a two-phase greedy approach — first guarantees 1 control per treated, then distributes remaining controls. This avoids forcing poor-quality matches when the control pool is limited.

Configurations: Fixed-K with k=1 and k=2, Varying-Ratio with max_k=2 and max_k=4. Comparing k=2 across both paradigms isolates the algorithm effect; comparing max_k=2 vs max_k=4 tests the bias/variance tradeoff of adding more controls.

#### 4. Matching Algorithms (`matching.py`)
The `Matcher` class implements two paradigms:

**Fixed-K Matching (Optimal)**:
- Uses the **Hungarian algorithm** (`scipy.linear_sum_assignment`) to minimize total propensity score distance
- Builds a cost matrix of shape `(n_treated * k, n_controls)` with propensity distances
- Pairs exceeding the caliper are masked to INF
- Guarantees exactly k controls per treated participant (or raises error)

**Varying-Ratio Matching (Two-Phase Greedy)**:
- **Phase 1**: Each treated participant gets exactly 1 closest control within caliper (skipped if none found)
- **Phase 2**: Remaining controls are greedily assigned to their closest treated participant, up to max_k per treated
- Populates `skipped_participants` for those who couldn't be matched

Both use **propensity score distance** with optional logit transform (configured via `IS_LOGIT`).

#### 5. Stratified Cox Proportional Hazards Analysis (`analysis.py`)
The `Analyzer` class fits a **Stratified Cox Proportional Hazards Model** (via `lifelines`):

- **Each matched set = one stratum** with its own baseline hazard
- **Why stratified**: Controls within a matched set are dependent (selected by propensity similarity). Standard Cox assumes independence. Stratification accounts for within-set dependence and allows each set to have a different baseline hazard.
- **Dynamic censoring**: Controls that transitioned to treated are censored at their surgery age in their old matched set (duration = surgery_age - match_time, event_observed = 0)
- **Output**: Hazard Ratio (HR), 95% CI, p-value. HR < 1 means treatment is protective.


#### 6. Visualizations (`visualizations.py`)

| Plot | What it shows |
|---|---|
| **Cancer Rate** | Grouped bar chart of cancer rate (%) for treated vs control per method |
| **Propensity Balance** | Before/after matching propensity score distributions (treated vs control) |
| **Avg Distance Lollipop** | Lollipop plot of average propensity distance before (random pairing) vs after matching per method |
| **Distance Distribution** | Boxplots of propensity distances between matched pairs, bucketed by score |
| **Kaplan-Meier** | Descriptive survival curves with risk tables (note: independence assumption not met) |
| **Cox-Adjusted Survival** | Model-based survival curves respecting stratification — the valid comparison |
| **Transitions Over Time** | Bar chart of control-to-treated transitions per age |
| **Controls Per Treated** | Histogram of how many controls each treated participant received |

**Why both KM and Cox-Adjusted curves**: The Kaplan-Meier curves are descriptive but violate the independence assumption (matched controls are dependent). The Cox-adjusted curves derive from the stratified model and are statistically valid. Comparing the two demonstrates why stratification matters — they can look quite different.

## Project Structure

```
MatchingResearch/
├── config.py              # Hyperparameters (N, k, caliper, etc.)
├── synthesize_data.py     # Synthetic data generation + EDA
├── participant.py         # Participant data class
├── experiment.py          # Dynamic matching orchestration + experiment configs
├── matching.py            # Matcher class (optimal + greedy algorithms)
├── analysis.py            # Stratified Cox PH analysis
├── visualizations.py      # All analysis plots (grid layout, N-method comparison)
├── .gitignore
├── logs/                  # Skipped participant CSVs (generated)
└── visualizations/        # Saved plot PNGs (generated)
```

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `NUM_PARTICIPANTS` | 10000 | Number of synthetic patients |
| `FIXED_K` | 2 | Controls per treated in fixed-k matching |
| `VARYING_RATIO_MAX_K` | 4 | Maximum controls in varying-ratio matching |
| `CALIPER` | 0.2 | Maximum allowed propensity distance for a match (logit scale) |
| `IS_LOGIT` | True | Use logit transform on propensity scores for distance |
| `CLIP_EPS` | 1e-6 | Clipping epsilon for logit transform |
| `INF` | 1e10 | Sentinel for invalid matches in cost matrix |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Experiment

```bash
python experiment.py
```

This will:
1. Generate 10,000 synthetic patients
2. Run all 4 matching configurations with dynamic transitions
3. Print Stratified Cox PH results for each
4. Save all visualizations to `visualizations/`
5. Log skipped participants to `logs/`

### Run EDA Only

```bash
python synthesize_data.py
```

Generates data and displays a grid of exploratory plots (distributions, propensity scores, etc.).

## Key Results (Typical Output)

With default settings (N=10000, caliper=0.2):
- **Hazard Ratio**: ~0.09-0.10 across all methods (treatment reduces cancer hazard by ~90%)
- **Fixed-K=2**: ~1750 matched sets, ~700 control-to-treated transitions
- **Varying-Ratio max=4**: ~1440 matched sets, ~1060 transitions, ~500 skipped
- All methods produce consistent HR estimates, validating robustness

## Key Design Decisions

1. **Propensity score = P(treatment | covariates)**, the standard definition. This estimates each patient's likelihood of receiving treatment given their covariates, ensuring that treated and control participants are matched on similar treatment probability profiles to reduce confounding.

2. **Covariate-dependent surgery probability**: Surgery likelihood is modulated by patient covariates via a logistic function: p = σ(base + β_f·F + β_x·X). This creates meaningful separation in propensity scores between treated and controls, making the matching step necessary and demonstrable.

3. **Stratified Cox PH over Kaplan-Meier**: KM assumes independence, which doesn't hold with 1:k matching. The stratified Cox model treats each matched set as a stratum with its own baseline hazard.

4. **Dynamic transitions with censoring**: When a matched control gets surgery, they are censored in their old matched set (outcome unknown from that point) and enter a new matched set as treated with fresh controls.

5. **Optimal matching for fixed-k, greedy for varying-ratio**: Hungarian algorithm guarantees global optimum for fixed-k. Two-phase greedy guarantees at least 1 control per treated before distributing extras.

6. **Comparing KM vs Cox-adjusted survival curves**: Kept both to demonstrate why the stratification choice matters — the curves differ significantly, validating that raw KM is misleading for 1:k matched data.

## References

1. Dagan, N., et al. (2021). BNT162b2 mRNA Covid-19 Vaccine in a Nationwide Mass Vaccination Setting. *NEJM*, 384(15), 1412-1423.
2. Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1.
3. Rosenbaum, P. R. *Design of Observational Studies*.
4. Yoo, J., et al. Stratified Cox Proportional Hazards Model for matched data.
