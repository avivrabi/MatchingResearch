"""Analysis visualizations for the matching experiment."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
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
    plot_kaplan_meier(analyzers)
    plot_cox_adjusted_survival(analyzers)
    plot_transitions_over_time(analyzers)
    plot_controls_per_treated(analyzers)


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
    pairs before matching (random pairing baseline) vs. after matching (actual pairs),
    for each method. Demonstrates that matching reduces propensity distance.
    """
    # Before matching: estimate average distance by randomly pairing treated with controls
    treated_all = [p for p in all_participants if p.is_treatment]
    control_all = [p for p in all_participants if not p.is_treatment]
    n_sample = min(len(treated_all), len(control_all), 5000)
    rng = np.random.default_rng(42)
    sampled_treated = rng.choice(treated_all, size=n_sample, replace=False)
    sampled_control = rng.choice(control_all, size=n_sample, replace=False)
    before_distances = [
        calculate_propensity_distance(t.propensity_score, c.propensity_score, IS_LOGIT)
        for t, c in zip(sampled_treated, sampled_control)
    ]
    before_avg = np.mean(before_distances)

    # After matching: average distance across actual matched pairs per method
    method_names = []
    after_avgs = []
    for analyzer in analyzers:
        distances = []
        for match in analyzer.matches:
            treated = match["treated"]
            for control in match["control"]:
                distances.append(
                    calculate_propensity_distance(
                        treated.propensity_score, control.propensity_score, IS_LOGIT
                    )
                )
        method_names.append(analyzer.method_name)
        after_avgs.append(np.mean(distances))

    # Build lollipop plot
    fig, ax = plt.subplots(figsize=(max(8, len(method_names) * 2.5), 5))
    x = np.arange(len(method_names))

    BEFORE_COLOR = "#888888"
    AFTER_COLOR = TREATMENT_COLOR

    # Lollipop stems (vertical lines from after to before)
    for i in range(len(method_names)):
        ax.plot([x[i], x[i]], [after_avgs[i], before_avg], color="#CCCCCC",
                linewidth=2, zorder=1)

    # Before matching dots (same baseline for all methods)
    ax.scatter(x, [before_avg] * len(x), color=BEFORE_COLOR, s=100, zorder=2,
               label="Before Matching (random pairing)", edgecolors="white", linewidths=1.5)

    # After matching dots
    ax.scatter(x, after_avgs, color=AFTER_COLOR, s=100, zorder=2,
               label="After Matching", edgecolors="white", linewidths=1.5)

    # Value annotations
    for i in range(len(method_names)):
        ax.text(x[i] + 0.15, after_avgs[i], f"{after_avgs[i]:.3f}", fontsize=9,
                va="center", color=AFTER_COLOR)
    ax.text(x[-1] + 0.15, before_avg, f"{before_avg:.3f}", fontsize=9,
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
# 4. Kaplan-Meier Survival Curves (Figure 5 style)
# =====================================================================
def plot_kaplan_meier(analyzers):
    """
    Kaplan-Meier survival curves with risk tables for treated vs. controls.
    One subplot per method, with both arms overlaid.
    """
    n = len(analyzers)
    fig, axes = _create_grid(n, cell_width=7, cell_height=6)

    for ax, analyzer in zip(axes, analyzers):
        df = analyzer.df

        treated_df = df[df["treatment"] == 1]
        control_df = df[df["treatment"] == 0]

        kmf_treated = KaplanMeierFitter()
        kmf_treated.fit(treated_df["duration"], treated_df["event_observed"], label="Treatment")
        kmf_treated.plot_survival_function(ax=ax, color=TREATMENT_COLOR, ci_show=True)

        kmf_control = KaplanMeierFitter()
        kmf_control.fit(control_df["duration"], control_df["event_observed"], label="Control")
        kmf_control.plot_survival_function(ax=ax, color=CONTROL_COLOR, ci_show=True)

        ax.set_title(f"Survival Curve\n{analyzer.method_name}", fontsize=12)
        ax.set_xlabel("Years Since Matching")
        ax.set_ylabel("Survival Probability (Cancer-Free)")
        ax.legend(loc="lower left")

        # Risk table below the plot
        _add_risk_table(ax, kmf_treated, kmf_control)

    # Shared axes: same x and y range across all KM plots
    max_x = max(ax.get_xlim()[1] for ax in axes)
    min_y = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, max_x)
        ax.set_ylim(min_y, 1.05)

    fig.suptitle("Kaplan-Meier Survival Curves", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "kaplan_meier.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _add_risk_table(ax, kmf_treated, kmf_control):
    """Adds a risk table annotation below the KM plot."""
    # Pick time points for the risk table
    max_time = max(kmf_treated.timeline.max(), kmf_control.timeline.max())
    time_points = np.linspace(0, max_time, min(6, int(max_time) + 1)).astype(int)
    time_points = sorted(set(time_points))

    treated_at_risk = []
    control_at_risk = []
    for t in time_points:
        treated_at_risk.append(_at_risk_count(kmf_treated, t))
        control_at_risk.append(_at_risk_count(kmf_control, t))

    table_text = "At risk\n"
    table_text += "Treated:  " + "  ".join(f"{r:>5}" for r in treated_at_risk) + "\n"
    table_text += "Control:  " + "  ".join(f"{r:>5}" for r in control_at_risk)

    ax.text(0.02, -0.22, table_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))


def _at_risk_count(kmf, t):
    """Returns the number of subjects at risk at time t."""
    event_table = kmf.event_table
    valid = event_table.index[event_table.index <= t]
    if len(valid) == 0:
        return event_table["at_risk"].iloc[0] if len(event_table) > 0 else 0
    return int(event_table.loc[valid[-1], "at_risk"])

# =====================================================================
# 5. Cox-Adjusted Survival Curves
# =====================================================================
def plot_cox_adjusted_survival(analyzers):
    """
    Adjusted survival curves derived from the Stratified Cox PH model.
    Unlike raw KM curves, these account for the matched-set stratification.
    Uses the average baseline cumulative hazard across all strata, then applies
    the Cox treatment coefficient to produce adjusted curves for treated vs. control.
    S(t|treatment) = S0(t) ^ exp(beta * treatment)
    """
    n = len(analyzers)
    fig, axes = _create_grid(n, cell_width=7, cell_height=6)

    for ax, analyzer in zip(axes, analyzers):
        if analyzer.cox_model is None:
            analyzer.run_stratified_cox()

        # Get the baseline cumulative hazard per stratum, then average across strata
        baseline_ch = analyzer.cox_model.baseline_cumulative_hazard_
        # Average across all strata columns to get a single baseline
        avg_baseline_ch = baseline_ch.mean(axis=1).sort_index()

        # S0(t) = exp(-H0(t))
        baseline_survival = np.exp(-avg_baseline_ch)

        # Treatment coefficient
        beta = analyzer.cox_model.params_["treatment"]

        # S(t|treated) = S0(t)^exp(beta*1), S(t|control) = S0(t)^exp(beta*0) = S0(t)
        surv_control = baseline_survival
        surv_treated = baseline_survival ** np.exp(beta)

        ax.plot(surv_treated.index, surv_treated.values, color=TREATMENT_COLOR,
                linewidth=2, label="Treatment (adjusted)")
        ax.plot(surv_control.index, surv_control.values, color=CONTROL_COLOR,
                linewidth=2, label="Control (adjusted)")

        ax.set_title(f"Cox-Adjusted Survival\n{analyzer.method_name}", fontsize=12)
        ax.set_xlabel("Years Since Matching")
        ax.set_ylabel("Survival Probability (Cancer-Free)")
        ax.legend(loc="lower left")

    # Shared axes
    max_x = max(ax.get_xlim()[1] for ax in axes)
    min_y = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, max_x)
        ax.set_ylim(min_y, 1.05)

    fig.suptitle("Cox-Adjusted Survival Curves (Stratified)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cox_adjusted_survival.png"), dpi=150, bbox_inches="tight")
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
