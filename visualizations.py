"""Analysis visualizations for the matching experiment."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from matching import calculate_propensity_distance
from config import IS_LOGIT

# Output directory for all saved plots
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color Palette
TREATMENT_COLOR = "#4A90D9"  # Blue
CONTROL_COLOR = "#D96459"    # Red/salmon
METHOD_COLORS = ["#4A90D9", "#D96459", "#6AB87A", "#E5A84B", "#9B7FBF", "#E07BA0"]


def _grid_layout(n):
    """Returns (n_rows, n_cols) for a grid that fits n subplots."""
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _create_grid(n, cell_width=6, cell_height=5):
    """Creates a grid of subplots and returns (fig, flat list of axes, n_rows, n_cols)."""
    n_rows, n_cols = _grid_layout(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_width * n_cols, cell_height * n_rows))
    axes_flat = np.array(axes).flatten().tolist() if n > 1 else [axes]
    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat[:n]


def plot_all(analyzers, all_participants):
    """
    Main entry point: generates all analysis visualizations.

    Args:
        analyzers: list of Analyzer objects (one per matching method)
        all_participants: list of all Participant objects (for before-matching reference)
    """
    plot_cancer_rate(analyzers)
    plot_propensity_balance(analyzers, all_participants)
    plot_avg_distance_lollipop(analyzers, all_participants)
    plot_distance_distribution(analyzers)
    plot_survival_combined(analyzers)
    plot_survival_km_vs_true(analyzers)
    plot_transitions_over_time(analyzers)
    plot_controls_per_treated(analyzers)
    plot_variance_comparison(analyzers)


# =====================================================================
# 1. Cancer Rate by Arm
# =====================================================================
def plot_cancer_rate(analyzers):
    """
    Grouped bar chart showing cancer rate (%) for treated vs. control per method.
    Separates the outcome (cancer rate) from the design choice (matching ratio).
    """
    method_names = []
    treated_rates = []
    control_rates = []

    for analyzer in analyzers:
        df = analyzer.df
        treated = df[df["treatment"] == 1]
        control = df[df["treatment"] == 0]
        treated_rates.append(treated["event_observed"].mean() * 100)
        control_rates.append(control["event_observed"].mean() * 100)
        method_names.append(analyzer.method_name)

    x = np.arange(len(method_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(method_names) * 2.5), 5))
    bars_treated = ax.bar(x - width / 2, treated_rates, width, label="Treatment",
                          color=TREATMENT_COLOR, edgecolor="white")
    bars_control = ax.bar(x + width / 2, control_rates, width, label="Control",
                          color=CONTROL_COLOR, edgecolor="white")

    # Add percentage labels on bars
    for bar in bars_treated:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars_control:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title("Cancer Rate by Arm Across Methods", fontsize=14, fontweight="bold")
    ax.legend()

    # Shared y-axis starting from 0
    max_y = max(max(treated_rates), max(control_rates))
    ax.set_ylim(0, max_y * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cancer_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 2. Propensity Score Balance (Before vs. After Matching)
# =====================================================================
def plot_propensity_balance(analyzers, all_participants):
    """
    Overlaid histograms of propensity scores for treated vs. controls,
    shown before matching (full population) and after matching (per method).
    All subplots share the same x-axis range [0, 1], and y-axis is normalized density in [0, 1].
    """
    n = len(analyzers)
    fig, axes = _create_grid(n + 1)

    # Before matching: full population
    treated_scores = [p.propensity_score for p in all_participants if p.is_treatment]
    control_scores = [p.propensity_score for p in all_participants if not p.is_treatment]

    ax = axes[0]
    sns.histplot(treated_scores, kde=True, stat="proportion", label="Treated",
                 color=TREATMENT_COLOR, alpha=0.5, ax=ax, bins=30)
    sns.histplot(control_scores, kde=True, stat="proportion", label="Control",
                 color=CONTROL_COLOR, alpha=0.5, ax=ax, bins=30)
    ax.set_title("Before Matching\n(Full Population)", fontsize=12)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Proportion")
    ax.legend()

    # After matching: per method
    for i, analyzer in enumerate(analyzers):
        ax = axes[i + 1]
        matched_treated_scores = []
        matched_control_scores = []
        for match in analyzer.matches:
            matched_treated_scores.append(match["treated"].propensity_score)
            for c in match["control"]:
                matched_control_scores.append(c.propensity_score)

        sns.histplot(matched_treated_scores, kde=True, stat="proportion", label="Treated",
                     color=TREATMENT_COLOR, alpha=0.5, ax=ax, bins=30)
        sns.histplot(matched_control_scores, kde=True, stat="proportion", label="Control",
                     color=CONTROL_COLOR, alpha=0.5, ax=ax, bins=30)
        ax.set_title(f"After Matching\n{analyzer.method_name}", fontsize=12)
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Proportion")
        ax.legend()

    # Enforce the same x-axis [0, 1] and y-axis range across all subplots
    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max_y)

    fig.suptitle("Propensity Score Balance: Before vs After Matching", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "propensity_balance.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 3. Average Propensity Distance: Before vs. After Matching (Lollipop)
# =====================================================================
def plot_avg_distance_lollipop(analyzers, all_participants):
    """
    Lollipop plot showing average propensity score distance between treated-control
    pairs before matching (random pairing baseline) vs after matching (actual pairs),
    for each method. The "before" baseline is computed per method using that method's
    actual matched treated participants paired with random controls.
    """
    control_all = [p for p in all_participants if not p.is_treatment]
    rng = np.random.default_rng(42)

    method_names = []
    before_avgs = []
    after_avgs = []

    for analyzer in analyzers:
        # Collect matched treated participants and after-matching distances
        matched_treated = []
        after_distances = []
        for match in analyzer.matches:
            treated = match["treated"]
            matched_treated.append(treated)
            for control in match["control"]:
                after_distances.append(
                    calculate_propensity_distance(
                        treated.propensity_score, control.propensity_score, IS_LOGIT
                    )
                )

        # Before matching: pair each matched treated with a random control
        random_controls = rng.choice(control_all, size=len(matched_treated), replace=True)
        before_distances = [
            calculate_propensity_distance(t.propensity_score, c.propensity_score, IS_LOGIT)
            for t, c in zip(matched_treated, random_controls)
        ]

        method_names.append(analyzer.method_name)
        before_avgs.append(np.mean(before_distances))
        after_avgs.append(np.mean(after_distances))

    # Build lollipop plot
    fig, ax = plt.subplots(figsize=(max(8, len(method_names) * 2.5), 5))
    x = np.arange(len(method_names))

    BEFORE_COLOR = "#888888"
    AFTER_COLOR = TREATMENT_COLOR

    # Lollipop stems (vertical lines from after to before)
    for i in range(len(method_names)):
        ax.plot([x[i], x[i]], [after_avgs[i], before_avgs[i]], color="#CCCCCC",
                linewidth=2, zorder=1)

    # Before matching dots
    ax.scatter(x, before_avgs, color=BEFORE_COLOR, s=100, zorder=2,
               label="Before Matching (random pairing)", edgecolors="white", linewidths=1.5)

    # After matching dots
    ax.scatter(x, after_avgs, color=AFTER_COLOR, s=100, zorder=2,
               label="After Matching", edgecolors="white", linewidths=1.5)

    # Value annotations
    for i in range(len(method_names)):
        ax.text(x[i] + 0.15, after_avgs[i], f"{after_avgs[i]:.3f}", fontsize=9,
                va="center", color=AFTER_COLOR)
        ax.text(x[i] + 0.15, before_avgs[i], f"{before_avgs[i]:.3f}", fontsize=9,
                va="center", color=BEFORE_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Avg Propensity Score Distance (logit scale)" if IS_LOGIT else "Avg Propensity Score Distance")
    ax.set_title("Average Propensity Distance: Before vs After Matching", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_distance_lollipop.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 4. Distance Distribution Between Matched Pairs (Figure 3 style)
# =====================================================================
def plot_distance_distribution(analyzers):
    """
    Boxplot of propensity score distances between each treated-control pair,
    bucketed by propensity score intervals.
    """
    n = len(analyzers)
    fig, axes = _create_grid(n, cell_width=7)

    for ax, analyzer in zip(axes, analyzers):
        distances = []
        ps_buckets = []
        for match in analyzer.matches:
            treated = match["treated"]
            for control in match["control"]:
                dist = calculate_propensity_distance(
                    treated.propensity_score, control.propensity_score, IS_LOGIT
                )
                # Bucket by the treated participant's propensity score
                bucket = f"{int(treated.propensity_score * 10) / 10:.1f}-{int(treated.propensity_score * 10) / 10 + 0.1:.1f}"
                distances.append(dist)
                ps_buckets.append(bucket)

        df_dist = pd.DataFrame({"Distance": distances, "Propensity Score": ps_buckets})
        # Sort buckets
        bucket_order = sorted(df_dist["Propensity Score"].unique())
        sns.boxplot(x="Propensity Score", y="Distance", data=df_dist,
                    order=bucket_order, ax=ax, color=TREATMENT_COLOR,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
        ax.set_title(f"Match Distance Distribution\n{analyzer.method_name}", fontsize=12)
        ax.set_xlabel("Propensity Score Interval")
        ax.set_ylabel("Distance Between Matched Pairs")
        ax.tick_params(axis="x", rotation=45)

    # Shared y-axis range
    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_y)

    fig.suptitle("Distance Distribution Between Matched Patients by Propensity Score",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distance_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _sync_axes(axes):

    """Synchronizes x and y axis ranges across a list of axes."""
    x_min = min(ax.get_xlim()[0] for ax in axes)
    x_max = max(ax.get_xlim()[1] for ax in axes)
    y_min = min(ax.get_ylim()[0] for ax in axes)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


# =====================================================================
# 4. Combined Survival Curves (KM + Cox Stratified + Cox Unstratified)
# =====================================================================
def plot_survival_combined(analyzers):
    """
    Combined survival plot: one subplot per method, each containing four
    curve types overlaid. Differentiated by line style:
      - Solid:     Kaplan-Meier (empirical, subject to censoring)
      - Dashed:    Cox-Adjusted Stratified (avg baseline across strata)
      - Dotted:    Cox-Adjusted Unstratified (single pooled baseline)
      - Dash-dot:  True Empirical (ground truth using actual cancer_age)
    Blue = Treatment, Red = Control.
    """
    n = len(analyzers)
    fig, axes = _create_grid(n, cell_width=8, cell_height=6)

    for ax, analyzer in zip(axes, analyzers):
        df = analyzer.df

        # --- 1. Kaplan-Meier (solid) ---
        treated_df = df[df["treatment"] == 1]
        control_df = df[df["treatment"] == 0]

        kmf_treated = KaplanMeierFitter()
        kmf_treated.fit(treated_df["duration"], treated_df["event_observed"])
        ax.plot(kmf_treated.survival_function_.index,
                kmf_treated.survival_function_.values,
                color=TREATMENT_COLOR, linewidth=2, linestyle="-", label="Treatment (KM)")

        kmf_control = KaplanMeierFitter()
        kmf_control.fit(control_df["duration"], control_df["event_observed"])
        ax.plot(kmf_control.survival_function_.index,
                kmf_control.survival_function_.values,
                color=CONTROL_COLOR, linewidth=2, linestyle="-", label="Control (KM)")

        # --- 2. Cox Stratified (dashed) ---
        if analyzer.cox_model is None:
            analyzer.run_stratified_cox()

        baseline_ch = analyzer.cox_model.baseline_cumulative_hazard_
        avg_baseline_ch = baseline_ch.mean(axis=1).sort_index()
        baseline_surv_strat = np.exp(-avg_baseline_ch)
        beta_strat = analyzer.cox_model.params_["treatment"]

        ax.plot(baseline_surv_strat.index,
                (baseline_surv_strat ** np.exp(beta_strat)).values,
                color=TREATMENT_COLOR, linewidth=2, linestyle="--", label="Treatment (Cox Stratified)")
        ax.plot(baseline_surv_strat.index,
                baseline_surv_strat.values,
                color=CONTROL_COLOR, linewidth=2, linestyle="--", label="Control (Cox Stratified)")

        # --- 3. Cox Unstratified (dotted) ---
        cox_unstrat = CoxPHFitter()
        df_no_strata = df[["duration", "event_observed", "treatment"]].copy()
        cox_unstrat.fit(df_no_strata, duration_col="duration", event_col="event_observed")

        baseline_ch_unstrat = cox_unstrat.baseline_cumulative_hazard_.iloc[:, 0].sort_index()
        baseline_surv_unstrat = np.exp(-baseline_ch_unstrat)
        beta_unstrat = cox_unstrat.params_["treatment"]

        ax.plot(baseline_surv_unstrat.index,
                (baseline_surv_unstrat ** np.exp(beta_unstrat)).values,
                color=TREATMENT_COLOR, linewidth=2, linestyle=":", label="Treatment (Cox Unstratified)")
        ax.plot(baseline_surv_unstrat.index,
                baseline_surv_unstrat.values,
                color=CONTROL_COLOR, linewidth=2, linestyle=":", label="Control (Cox Unstratified)")

        # --- 4. True Empirical Survival (dash-dot) ---
        # Direct computation from actual cancer_age — no model, no estimator.
        # Transitioning controls are censored at surgery_age (their post-surgery
        # cancer_age is contaminated by the treatment effect).
        true_treated_times = []
        true_control_times = []

        for match in analyzer.matches:
            match_time = match.get("match_time", 0)
            true_treated_times.append(match["treated"].cancer_age - match_time)
            for control in match["control"]:
                censor_age = analyzer.transitioned_controls.get(control.id, None)
                if censor_age is not None:
                    true_control_times.append(censor_age - match_time)
                else:
                    true_control_times.append(control.cancer_age - match_time)

        true_treated_times = np.array(true_treated_times)
        true_control_times = np.array(true_control_times)

        for times, color, label in [
            (true_treated_times, TREATMENT_COLOR, "Treatment (True)"),
            (true_control_times, CONTROL_COLOR, "Control (True)"),
        ]:
            sorted_t = np.sort(np.unique(times))
            survival = np.array([np.mean(times > t) for t in sorted_t])
            ax.plot(sorted_t, survival, color=color, linewidth=1.5,
                    linestyle="-.", alpha=0.7, label=label)

        ax.set_title(f"{analyzer.method_name}", fontsize=12)
        ax.set_xlabel("Years Since Matching")
        ax.set_ylabel("Survival Probability (Cancer-Free)")
        ax.legend(loc="lower left", fontsize=7)

    # Shared axes
    max_x = max(ax.get_xlim()[1] for ax in axes)
    min_y = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, max_x)
        ax.set_ylim(min_y, 1.05)

    fig.suptitle("Survival Curves: KM vs Cox Stratified vs Cox Unstratified",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "survival_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 5. KM vs True Empirical Survival Curves
# =====================================================================
def plot_survival_km_vs_true(analyzers):
    """
    Simplified survival plot: one subplot per method, each containing only
    KM and True Empirical curves. Shows how well KM (descriptive) recovers
    the ground-truth survival without any model.
      - Solid:     Kaplan-Meier (empirical, subject to censoring)
      - Dash-dot:  True Empirical (ground truth using actual cancer_age)
    Blue = Treatment, Red = Control.
    """
    n = len(analyzers)
    fig, axes = _create_grid(n, cell_width=8, cell_height=6)

    for ax, analyzer in zip(axes, analyzers):
        df = analyzer.df

        # --- KM (solid) ---
        treated_df = df[df["treatment"] == 1]
        control_df = df[df["treatment"] == 0]

        kmf_treated = KaplanMeierFitter()
        kmf_treated.fit(treated_df["duration"], treated_df["event_observed"])
        ax.plot(kmf_treated.survival_function_.index,
                kmf_treated.survival_function_.values,
                color=TREATMENT_COLOR, linewidth=2, linestyle="-", label="Treatment (KM)")

        kmf_control = KaplanMeierFitter()
        kmf_control.fit(control_df["duration"], control_df["event_observed"])
        ax.plot(kmf_control.survival_function_.index,
                kmf_control.survival_function_.values,
                color=CONTROL_COLOR, linewidth=2, linestyle="-", label="Control (KM)")

        # --- True Empirical (dash-dot) ---
        true_treated_times = []
        true_control_times = []

        for match in analyzer.matches:
            match_time = match.get("match_time", 0)
            true_treated_times.append(match["treated"].cancer_age - match_time)
            for control in match["control"]:
                censor_age = analyzer.transitioned_controls.get(control.id, None)
                if censor_age is not None:
                    true_control_times.append(censor_age - match_time)
                else:
                    true_control_times.append(control.cancer_age - match_time)

        true_treated_times = np.array(true_treated_times)
        true_control_times = np.array(true_control_times)

        for times, color, label in [
            (true_treated_times, TREATMENT_COLOR, "Treatment (True)"),
            (true_control_times, CONTROL_COLOR, "Control (True)"),
        ]:
            sorted_t = np.sort(np.unique(times))
            survival = np.array([np.mean(times > t) for t in sorted_t])
            ax.plot(sorted_t, survival, color=color, linewidth=1.5,
                    linestyle="-.", alpha=0.7, label=label)

        ax.set_title(f"{analyzer.method_name}", fontsize=12)
        ax.set_xlabel("Years Since Matching")
        ax.set_ylabel("Survival Probability (Cancer-Free)")
        ax.legend(loc="lower left", fontsize=9)

    # Shared axes
    max_x = max(ax.get_xlim()[1] for ax in axes)
    min_y = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, max_x)
        ax.set_ylim(min_y, 1.05)

    fig.suptitle("Survival Curves: KM vs True Empirical",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "survival_km_vs_true.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 6. Transitions Over Time
# =====================================================================
def plot_transitions_over_time(analyzers):
    """
    Bar chart showing how many controls transitioned to treated at each surgery age,
    for each matching method.
    """
    n = len(analyzers)
    fig, axes = _create_grid(n, cell_width=7)

    for ax, analyzer in zip(axes, analyzers):
        transition_ages = list(analyzer.transitioned_controls.values())
        if not transition_ages:
            ax.text(0.5, 0.5, "No transitions", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Transitions Over Time\n{analyzer.method_name}")
            continue

        age_counts = pd.Series(transition_ages).value_counts().sort_index()
        ax.bar(age_counts.index, age_counts.values, color=TREATMENT_COLOR, edgecolor="white", alpha=0.8)
        ax.set_xlabel("Surgery Age (Transition Time)")
        ax.set_ylabel("Number of Transitions")
        ax.set_title(f"Control-to-Treated Transitions\n{analyzer.method_name}", fontsize=12)

        # Add total count annotation
        total = len(transition_ages)
        ax.text(0.95, 0.95, f"Total: {total}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    # Shared axes across all subplots
    _sync_axes(axes)

    fig.suptitle("Dynamic Transitions Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "transitions_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 6. Controls Per Treated Distribution
# =====================================================================
def plot_controls_per_treated(analyzers):
    """
    Histogram showing the number of controls matched to each treated participant,
    for each method. Useful for comparing fixed-k vs. varying-ratio.
    """
    n = len(analyzers)
    fig, axes = _create_grid(n)

    for ax, analyzer in zip(axes, analyzers):
        controls_per_treated = [len(m["control"]) for m in analyzer.matches]
        bins = range(0, max(controls_per_treated) + 2)
        ax.hist(controls_per_treated, bins=bins, color=TREATMENT_COLOR,
                edgecolor="white", alpha=0.8, align="left")
        ax.set_xlabel("Number of Controls")
        ax.set_ylabel("Number of Treated Participants")
        ax.set_title(f"Controls Per Treated\n{analyzer.method_name}", fontsize=12)
        ax.set_xticks(range(0, max(controls_per_treated) + 1))

        # Stats annotation
        mean_k = np.mean(controls_per_treated)
        ax.text(0.95, 0.95, f"Mean: {mean_k:.2f}\nk={analyzer.k}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    # Shared y-axis across all subplots
    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_y)

    fig.suptitle("Controls Per Treated Participant", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "controls_per_treated.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 8. Variance Comparison Across Methods
# =====================================================================
def plot_variance_comparison(analyzers):
    """
    Visualization comparing variance metrics (SE and CI width) across matching methods.
    Creates a multi-panel plot showing:
    1. Standard Error of coefficient (lower is better)
    2. CI width of hazard ratio (lower is better)
    3. Forest plot of hazard ratios with confidence intervals
    """
    # Extract variance metrics
    variance_metrics = [analyzer.get_variance_metrics() for analyzer in analyzers]
    df = pd.DataFrame(variance_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Standard Error comparison (bar chart)
    ax = axes[0]
    x = np.arange(len(df))
    bars = ax.bar(x, df["coef_se"], color=METHOD_COLORS[:len(df)],
                  edgecolor="white", alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df["coef_se"])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Standard Error of Coefficient")
    ax.set_title("Coefficient Standard Error\n(Lower = More Precise)", fontweight="bold")
    ax.set_ylim(0, df["coef_se"].max() * 1.15)

    # Panel 2: CI Width comparison (bar chart)
    ax = axes[1]
    bars = ax.bar(x, df["ci_width"], color=METHOD_COLORS[:len(df)],
                  edgecolor="white", alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df["ci_width"])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("95% CI Width (HR)")
    ax.set_title("Confidence Interval Width\n(Lower = More Precise)", fontweight="bold")
    ax.set_ylim(0, df["ci_width"].max() * 1.15)

    # Panel 3: Forest plot (Hazard Ratios with CIs)
    ax = axes[2]
    y_pos = np.arange(len(df))

    # Plot horizontal lines for CIs
    for i, row in df.iterrows():
        ax.plot([row["hr_lower_95"], row["hr_upper_95"]], [i, i],
                color=METHOD_COLORS[i], linewidth=2, marker='|', markersize=10)
        # Plot point estimate
        ax.plot(row["hazard_ratio"], i, 'o', color=METHOD_COLORS[i],
                markersize=10, markeredgecolor="white", markeredgewidth=1.5)

    # Add a vertical line at HR=1 (no effect)
    ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(1, len(df) - 0.5, 'No Effect', ha='center', va='bottom',
            fontsize=9, color='gray', rotation=90)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["method"], fontsize=9)
    ax.set_xlabel("Hazard Ratio")
    ax.set_title("Hazard Ratios with 95% CI\n(Forest Plot)", fontweight="bold")
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Add HR values as text
    for i, row in df.iterrows():
        ax.text(row["hr_upper_95"] + 0.05, i,
                f"HR={row['hazard_ratio']:.3f} [{row['hr_lower_95']:.3f}-{row['hr_upper_95']:.3f}]",
                va='center', fontsize=8, color=METHOD_COLORS[i])

    fig.suptitle("Variance Comparison Across Matching Methods",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "variance_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
