"""
Statistical analysis and visualization of ConDOR ablation study results.

Compares RMSE across study configurations with pairwise t-tests and ANOVA,
generates bar charts with significance annotations, and produces summary
tables. Also plots QoL response distributions, stage/resectability
distributions, and treatment timing from the underlying PACAP/NKR data.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

STUDY_MAPPING = {
    'Baseline': 'study_baseline',
    'RDM with Clinical Features': 'study_rdm_with_clinical_features',
    'RDM with Mortality Risk': 'study_rdm_with_mortality_risk',
    'RDM Clinical Features + Mortality Risk': 'study_rdm_clinical_features__mortality_risk',
    'TDM with Treatment': 'study_tdm_with_treatment',
    'RDM Clinical Features + TDM Treatment': 'study_rdm_clinical_features__tdm_treatment',
    'Full Model (All Features)': 'study_full_model_(all_features)',
}

try:
    from xai_decision_making import config as main_config
    ID_COL = getattr(main_config, 'KEY_NK', 'key_nkr')
    TIME_COL = getattr(main_config, 'PROMS_DATE_COL', 'proms_dat_int')
except Exception:
    ID_COL = 'key_nkr'
    TIME_COL = 'proms_dat_int'

def load_all_results(eval_set='Black_Envelope'):
    results = {}
    base_dir = "outputs"
    if not os.path.exists(base_dir):
        print(f"✗ Critical: '{base_dir}' directory not found.")
        return {}

    print(f"Searching for results in {base_dir}...")

    for pretty_name, folder_name in STUDY_MAPPING.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            continue

        csv_files = glob.glob(f"{folder_path}/results_*.csv")
        if not csv_files:
            print(f"~ Folder exists but no results CSV for: {pretty_name}")
            continue

        df = pd.read_csv(csv_files[0])

        if 'eval_set' in df.columns:
            if eval_set == 'all':
                print(f"✓ Loaded {pretty_name}: {len(df)} rows (all eval sets)")
            else:
                df_filtered = df[df['eval_set'] == eval_set].copy()
                if len(df_filtered) > 0:
                    df = df_filtered
                    print(f"✓ Loaded {pretty_name}: {len(df)} folds ({eval_set})")
                else:
                    print(f"~ {pretty_name}: No {eval_set} results found, skipping")
                    continue
        else:
            print(f"✓ Loaded {pretty_name}: {len(df)} runs (no eval_set column)")

        results[pretty_name] = df

    return results

def perform_statistical_tests(results, eval_set_name=""):
    print("\n" + "=" * 80)
    print(f"STATISTICAL ANALYSIS - {eval_set_name}")
    print("=" * 80)

    rdm_metrics = {}
    full_metrics = {}

    for name, df in results.items():
        if 'RDM_RMSE' in df.columns:
            rdm_metrics[name] = df['RDM_RMSE'].values
        if 'Full_RMSE' in df.columns:
            full_metrics[name] = df['Full_RMSE'].values

    if len(rdm_metrics) > 1:
        print("\n--- RDM RMSE Statistical Tests ---")
        rdm_df = pd.DataFrame({k: pd.Series(v) for k, v in rdm_metrics.items()})
        print(rdm_df.describe().round(4))

        f_stat, pval_anova = stats.f_oneway(*[v for v in rdm_metrics.values() if len(v) > 0])
        print(f"\nOne-Way ANOVA (RDM RMSE): F={f_stat:.3f}, p={pval_anova:.4f}")

        print("\nPairwise T-Tests (RDM RMSE):")
        methods = list(rdm_metrics.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                if len(rdm_metrics[m1]) < 2 or len(rdm_metrics[m2]) < 2:
                    continue
                stat, pval = stats.ttest_ind(rdm_metrics[m1], rdm_metrics[m2], equal_var=False)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                diff = np.mean(rdm_metrics[m1]) - np.mean(rdm_metrics[m2])
                print(f"  {m1} vs {m2}: Diff={diff:.4f} | p={pval:.4f} {sig}")

    if len(full_metrics) > 1:
        print("\n--- Full Trajectory RMSE Statistical Tests ---")
        full_df = pd.DataFrame({k: pd.Series(v) for k, v in full_metrics.items()})
        print(full_df.describe().round(4))

        f_stat, pval_anova = stats.f_oneway(*[v for v in full_metrics.values() if len(v) > 0])
        print(f"\nOne-Way ANOVA (Full RMSE): F={f_stat:.3f}, p={pval_anova:.4f}")

        print("\nPairwise T-Tests (Full RMSE):")
        methods = list(full_metrics.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                if len(full_metrics[m1]) < 2 or len(full_metrics[m2]) < 2:
                    continue
                stat, pval = stats.ttest_ind(full_metrics[m1], full_metrics[m2], equal_var=False)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                diff = np.mean(full_metrics[m1]) - np.mean(full_metrics[m2])
                print(f"  {m1} vs {m2}: Diff={diff:.4f} | p={pval:.4f} {sig}")

    return rdm_metrics, full_metrics

def compare_baseline_vs_rdm_tdm(results, eval_set_name=""):
    baseline_name = "Baseline"
    rdm_tdm_name = "RDM Clinical Features + TDM Treatment"

    if baseline_name not in results or rdm_tdm_name not in results:
        print(f"\n⚠ Cannot compare: {baseline_name} or {rdm_tdm_name} not found in results")
        return

    print(f"\n{'=' * 80}")
    print(f"SPECIFIC COMPARISON: {baseline_name} vs {rdm_tdm_name} ({eval_set_name})")
    print(f"{'=' * 80}")

    baseline_df = results[baseline_name]
    rdm_tdm_df = results[rdm_tdm_name]

    metrics_to_compare = ['RDM_RMSE', 'Full_RMSE']

    all_month_metrics = []
    for col in list(baseline_df.columns) + list(rdm_tdm_df.columns):
        if col.startswith('RMSE_') and col.endswith('M') and col not in all_month_metrics:
            if col in baseline_df.columns and col in rdm_tdm_df.columns:
                all_month_metrics.append(col)

    all_month_metrics = sorted(all_month_metrics, key=lambda c: int(c.replace('RMSE_', '').replace('M', '')) if c.replace('RMSE_', '').replace('M', '').isdigit() else 0)
    metrics_to_compare.extend(all_month_metrics)

    comparison_data = []
    for metric in metrics_to_compare:
        if metric not in baseline_df.columns or metric not in rdm_tdm_df.columns:
            continue

        b_vals = baseline_df[metric].dropna().values
        r_vals = rdm_tdm_df[metric].dropna().values

        if len(b_vals) < 2 or len(r_vals) < 2:
            print(f"\n{metric}: Insufficient data")
            continue

        b_mean, b_std = np.mean(b_vals), np.std(b_vals)
        r_mean, r_std = np.mean(r_vals), np.std(r_vals)
        stat, pval = stats.ttest_ind(b_vals, r_vals, equal_var=False)
        diff = b_mean - r_mean

        if pval < 0.001:
            sig = "*** (p < 0.001)"
            sig_sym = "***"
        elif pval < 0.01:
            sig = "** (p < 0.01)"
            sig_sym = "**"
        elif pval < 0.05:
            sig = "* (p < 0.05)"
            sig_sym = "*"
        else:
            sig = "ns (not significant)"
            sig_sym = "ns"

        print(f"\n{metric}:")
        print(f"  Baseline: {b_mean:.3f} ± {b_std:.3f} (n={len(b_vals)})")
        print(f"  RDM+TDM:  {r_mean:.3f} ± {r_std:.3f} (n={len(r_vals)})")
        print(f"  Diff: {diff:+.3f} ({'better' if diff > 0 else 'worse' if diff < 0 else 'same'}) | p={pval:.4f} {sig}")

        comparison_data.append({
            'Metric': metric,
            'Baseline_Mean': b_mean, 'Baseline_Std': b_std,
            'RDM_TDM_Mean': r_mean, 'RDM_TDM_Std': r_std,
            'Difference': diff, 'P_value': pval,
            'Significance': sig_sym,
            'N_Baseline': len(b_vals), 'N_RDM_TDM': len(r_vals),
        })

    if comparison_data:
        df_cmp = pd.DataFrame(comparison_data)
        display_df = df_cmp.copy()
        display_df['Baseline'] = display_df.apply(lambda x: f"{x['Baseline_Mean']:.3f} ± {x['Baseline_Std']:.3f}", axis=1)
        display_df['RDM_TDM'] = display_df.apply(lambda x: f"{x['RDM_TDM_Mean']:.3f} ± {x['RDM_TDM_Std']:.3f}", axis=1)
        display_df['Difference'] = display_df['Difference'].apply(lambda x: f"{x:+.3f}")
        display_df['P_value'] = display_df['P_value'].apply(lambda x: f"{x:.4f}")
        print(f"\n{'=' * 80}")
        print("SUMMARY TABLE")
        print(f"{'=' * 80}\n")
        print(display_df[['Metric', 'Baseline', 'RDM_TDM', 'Difference', 'P_value', 'Significance']].to_string(index=False))

        os.makedirs("outputs/analysis_tables", exist_ok=True)
        safe = eval_set_name.lower().replace(' ', '_').replace('(', '').replace(')', '') if eval_set_name else 'all'
        csv_path = f'outputs/analysis_tables/baseline_vs_rdm_tdm_{safe}.csv'
        df_cmp.to_csv(csv_path, index=False)
        print(f"\n✓ Saved comparison table to {csv_path}")

def create_summary_table(results, eval_set_name=""):
    if not results:
        return None

    all_months = set()
    for df in results.values():
        for col in df.columns:
            if col.startswith('RMSE_') and col.endswith('M'):
                try:
                    all_months.add(int(col.replace('RMSE_', '').replace('M', '')))
                except ValueError:
                    pass
    all_months = sorted(all_months)

    table_data = []
    for name, df in results.items():
        row = {
            'Condition': name,
            'RDM RMSE (Mean ± Std)': f"{df['RDM_RMSE'].mean():.3f} ± {df['RDM_RMSE'].std():.3f}",
            'RDM RMSE Mean': df['RDM_RMSE'].mean(),
            'RDM RMSE Std': df['RDM_RMSE'].std(),
            'Full RMSE (Mean ± Std)': f"{df['Full_RMSE'].mean():.3f} ± {df['Full_RMSE'].std():.3f}",
            'Full RMSE Mean': df['Full_RMSE'].mean(),
            'Full RMSE Std': df['Full_RMSE'].std(),
        }
        for month in all_months:
            col = f'RMSE_{month}M'
            if col in df.columns:
                m, s = df[col].mean(), df[col].std()
                row[f'RMSE {month}M (Mean ± Std)'] = f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else "N/A"
                row[f'RMSE {month}M Mean'] = m
                row[f'RMSE {month}M Std'] = s
        table_data.append(row)

    df_table = pd.DataFrame(table_data)
    ordered = list(STUDY_MAPPING.keys())
    df_table['Order'] = df_table['Condition'].apply(lambda x: ordered.index(x) if x in ordered else 999)
    df_table = df_table.sort_values('Order').drop('Order', axis=1)

    print(f"\n{'=' * 100}")
    print(f"SUMMARY TABLE - {eval_set_name}")
    print(f"{'=' * 100}\n")

    display_cols = ['Condition', 'RDM RMSE (Mean ± Std)', 'Full RMSE (Mean ± Std)']
    for month in all_months:
        col = f'RMSE {month}M (Mean ± Std)'
        if col in df_table.columns:
            display_cols.append(col)
    print(df_table[display_cols].to_string(index=False))

    os.makedirs("outputs/analysis_tables", exist_ok=True)
    safe = eval_set_name.lower().replace(' ', '_').replace('(', '').replace(')', '') if eval_set_name else 'all'
    csv_path = f'outputs/analysis_tables/summary_table_{safe}.csv'
    csv_cols = ['Condition', 'RDM RMSE Mean', 'RDM RMSE Std', 'Full RMSE Mean', 'Full RMSE Std']
    for month in all_months:
        for suffix in ['Mean', 'Std']:
            c = f'RMSE {month}M {suffix}'
            if c in df_table.columns:
                csv_cols.append(c)
    df_table[csv_cols].to_csv(csv_path, index=False)
    print(f"\n✓ Saved summary table to {csv_path}")
    return df_table

def plot_summary(results, eval_set_name="", rdm_metrics=None, full_metrics=None):
    if not results:
        return

    names, means_rdm, stds_rdm, means_full, stds_full = [], [], [], [], []
    for name, df in results.items():
        names.append(name)
        means_rdm.append(df['RDM_RMSE'].mean())
        stds_rdm.append(df['RDM_RMSE'].std())
        means_full.append(df['Full_RMSE'].mean())
        stds_full.append(df['Full_RMSE'].std())

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, means_rdm, width, yerr=stds_rdm, label='RDM RMSE', alpha=0.8)
    ax.bar(x + width / 2, means_full, width, yerr=stds_full, label='Full RMSE', alpha=0.8)

    title = 'Ablation Study - Black Envelope Test Set (Lower is Better)' if eval_set_name == "Black Envelope" else f'Study Comparison - {eval_set_name} (Lower is Better)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)

    main_legend = ax.legend(['RDM RMSE', 'Full RMSE'], fontsize=10, loc='upper left', framealpha=0.9)
    ax.add_artist(main_legend)
    ax.grid(axis='y', alpha=0.3)

    if rdm_metrics is not None and full_metrics is not None:
        max_val = max(max(means_rdm) + max(stds_rdm), max(means_full) + max(stds_full))
        y_offset = max_val * 0.05
        name_to_idx = {name: idx for idx, name in enumerate(names)}

        sig_pairs = []
        methods = list(full_metrics.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                if len(full_metrics[m1]) < 2 or len(full_metrics[m2]) < 2:
                    continue
                if m1 not in name_to_idx or m2 not in name_to_idx:
                    continue
                stat, pval = stats.ttest_ind(full_metrics[m1], full_metrics[m2], equal_var=False)
                if pval < 0.05:
                    sig_pairs.append((name_to_idx[m1], name_to_idx[m2], pval))

        if sig_pairs:
            sig_pairs_sorted = sorted(sig_pairs, key=lambda p: p[2])
            current_y = max_val + y_offset * 1.5
            line_height = y_offset * 0.4

            for idx1, idx2, pval in sig_pairs_sorted:
                if pval < 0.001:
                    sig_text, sig_color, lw = "***", '#e74c3c', 2.5
                elif pval < 0.01:
                    sig_text, sig_color, lw = "**", '#f39c12', 2.0
                else:
                    sig_text, sig_color, lw = "*", '#3498db', 1.8

                x1 = x[idx1] + width / 2
                x2 = x[idx2] + width / 2
                ax.plot([x1, x2], [current_y, current_y], color=sig_color, linewidth=lw, alpha=0.9, zorder=10)
                ax.plot([x1, x1], [current_y - line_height / 2, current_y], color=sig_color, linewidth=lw, alpha=0.9, zorder=10)
                ax.plot([x2, x2], [current_y - line_height / 2, current_y], color=sig_color, linewidth=lw, alpha=0.9, zorder=10)
                ax.text((x1 + x2) / 2, current_y + line_height / 3, sig_text,
                        ha='center', va='bottom', fontsize=12, fontweight='bold',
                        color=sig_color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=sig_color, linewidth=1.5), zorder=11)
                current_y += y_offset

            ax.set_ylim(top=current_y + y_offset)

            legend_elements = []
            if any(p[2] < 0.001 for p in sig_pairs_sorted):
                legend_elements.append(Patch(facecolor='#e74c3c', edgecolor='#e74c3c', label='*** p < 0.001'))
            if any(0.001 <= p[2] < 0.01 for p in sig_pairs_sorted):
                legend_elements.append(Patch(facecolor='#f39c12', edgecolor='#f39c12', label='** p < 0.01'))
            if any(0.01 <= p[2] < 0.05 for p in sig_pairs_sorted):
                legend_elements.append(Patch(facecolor='#3498db', edgecolor='#3498db', label='* p < 0.05'))
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    try:
        os.makedirs("outputs/analysis_plots", exist_ok=True)
        safe = eval_set_name.lower().replace(' ', '_').replace('(', '').replace(')', '') if eval_set_name else 'all'
        plt.tight_layout()
        plot_path = f'outputs/analysis_plots/comparison_bar_{safe}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n✗ Error saving plot: {e}")
        plt.close()

def plot_qol_responses_per_month(pacap_path='data/dev_set_pacap.csv', output_dir='outputs/analysis_plots'):
    if not os.path.exists(pacap_path):
        print(f"Warning: PACAP file not found at {pacap_path}. Skipping.")
        return

    print(f"\n{'=' * 80}\nPLOTTING QOL RESPONSES PER MONTH\n{'=' * 80}\n")
    df = pd.read_csv(pacap_path, low_memory=False)

    # Try to resolve time column
    time_col = TIME_COL
    if time_col not in df.columns:
        candidates = [c for c in df.columns if 'prom' in c.lower() and ('dat' in c.lower() or 'int' in c.lower())]
        if candidates:
            time_col = candidates[0]
        else:
            print(f"Warning: Time column not found. Skipping.")
            return

    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df['time_months'] = df[time_col] / 30.44
    df = df.dropna(subset=['time_months'])
    df = df[df['time_months'] >= 0]
    df['month_bin'] = df['time_months'].round().astype(int)

    if ID_COL in df.columns:
        month_counts = df.groupby('month_bin')[ID_COL].nunique().sort_index()
    else:
        month_counts = df['month_bin'].value_counts().sort_index()

    month_counts = month_counts[month_counts.index <= 60]

    if 'black_envelop' in pacap_path.lower():
        dataset_name, plot_filename = "Black Envelope Test Set", 'qol_responses_per_month_black_envelope.png'
    elif 'combined' in pacap_path.lower():
        dataset_name, plot_filename = "Dev + Test Set (Combined)", 'qol_responses_per_month_dev_test_combined.png'
    else:
        dataset_name, plot_filename = "Dev Set", 'qol_responses_per_month_dev_set.png'

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(month_counts.index, month_counts.values, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Months Since Diagnosis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients with QoL Response', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Patients with QoL Responses Over Time Since Diagnosis', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    max_month = month_counts.index.max()
    ax.set_xticks(range(0, int(max_month) + 1, 6))
    ax.set_xlim(-0.5, max_month + 0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved QoL response distribution plot ({dataset_name}) to {plot_path}")

    total_patients = df[ID_COL].nunique() if ID_COL in df.columns else len(df)
    print(f"\nSummary Statistics:")
    print(f"  Total unique patients: {total_patients:,}")
    print(f"  Total QoL responses: {len(df):,}")
    print(f"  Peak month: {month_counts.idxmax()} months ({month_counts.max()} patients)")
    plt.close()

def plot_responses_per_patient(pacap_path='data/dev_set_pacap.csv', output_dir='outputs/analysis_plots'):
    if not os.path.exists(pacap_path):
        print(f"Warning: PACAP file not found at {pacap_path}. Skipping.")
        return

    print(f"\n{'=' * 80}\nPLOTTING RESPONSES PER PATIENT\n{'=' * 80}\n")
    df = pd.read_csv(pacap_path, low_memory=False)

    if ID_COL not in df.columns:
        print(f"Warning: Patient ID column '{ID_COL}' not found. Skipping.")
        return

    responses_per_patient = df.groupby(ID_COL).size()
    response_counts = responses_per_patient.value_counts().sort_index()

    if 'black_envelop' in pacap_path.lower():
        plot_filename = 'responses_per_patient_black_envelope.png'
    elif 'combined' in pacap_path.lower():
        plot_filename = 'responses_per_patient_dev_test_combined.png'
    else:
        plot_filename = 'responses_per_patient_dev_set.png'

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(response_counts.index, response_counts.values, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Number of Responses per Patient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of QoL Responses per Patient', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    stats_text = (
        f"Total Patients: {len(responses_per_patient):,}\n"
        f"Total Responses: {responses_per_patient.sum():,}\n"
        f"Mean/Patient: {responses_per_patient.mean():.2f}\n"
        f"Median/Patient: {responses_per_patient.median():.1f}\n"
        f"Max: {responses_per_patient.max()}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xticks(response_counts.index)
    ax.set_xlim(-0.5, max(response_counts.index) + 0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved responses per patient plot to {plot_path}")

    print(f"\nSummary Statistics:")
    print(f"  Total patients: {len(responses_per_patient):,}")
    print(f"  Mean: {responses_per_patient.mean():.2f} | Median: {responses_per_patient.median():.1f}")
    print(f"  1 response: {(responses_per_patient == 1).sum()} | 2+: {(responses_per_patient >= 2).sum()} | 3+: {(responses_per_patient >= 3).sum()}")
    plt.close()

def plot_stage_distribution(clinical_paths=None, pacap_paths=None, output_dir='outputs/analysis_plots'):
    STAGE_COL = 'stadium'
    if clinical_paths is None:
        clinical_paths = ['data/dev_set.csv', 'data/test_set.csv']
    if pacap_paths is None:
        pacap_paths = ['data/dev_set_pacap.csv', 'data/test_set_pacap.csv']

    dfs_clin = [pd.read_csv(p, low_memory=False) for p in clinical_paths if os.path.exists(p)]
    dfs_pacap = [pd.read_csv(p, low_memory=False) for p in pacap_paths if os.path.exists(p)]

    if not dfs_clin or not dfs_pacap:
        print("Warning: Missing clinical or PACAP files. Skipping stage distribution plot.")
        return

    df_clinical = pd.concat(dfs_clin, ignore_index=True)
    if ID_COL in df_clinical.columns:
        df_clinical = df_clinical.drop_duplicates(subset=[ID_COL], keep='first')

    df_pacap = pd.concat(dfs_pacap, ignore_index=True)
    if ID_COL in df_pacap.columns:
        df_pacap = df_pacap.drop_duplicates(subset=[ID_COL], keep='first')

    print(f"\n{'=' * 80}\nPLOTTING STAGE DISTRIBUTION\n{'=' * 80}\n")

    if STAGE_COL not in df_clinical.columns:
        print(f"Warning: Stage column '{STAGE_COL}' not found. Skipping.")
        return

    pacap_ids = set(df_pacap[ID_COL].unique()) if ID_COL in df_pacap.columns else set()
    df_ncr_pacap = df_clinical[df_clinical[ID_COL].isin(pacap_ids)].copy() if ID_COL in df_clinical.columns else pd.DataFrame()

    def clean_stage(val):
        if pd.isna(val):
            return 'Unknown'
        s = str(val).strip().upper()
        if s in ['1', '1A', '1B', 'I', 'IA', 'IB']:
            return 'Stage I'
        if s in ['2', '2A', '2B', 'II', 'IIA', 'IIB']:
            return 'Stage II'
        if s in ['3', 'III']:
            return 'Stage III'
        if s in ['4', 'IV', 'M']:
            return 'Stage IV'
        return 'Unknown'

    df_clinical['stage_clean'] = df_clinical[STAGE_COL].apply(clean_stage)
    if len(df_ncr_pacap) > 0:
        df_ncr_pacap['stage_clean'] = df_ncr_pacap[STAGE_COL].apply(clean_stage)

    stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'Unknown']
    counts_all = df_clinical['stage_clean'].value_counts().reindex(stage_order, fill_value=0)
    counts_pacap = df_ncr_pacap['stage_clean'].value_counts().reindex(stage_order, fill_value=0) if len(df_ncr_pacap) > 0 else pd.Series([0] * 5, index=stage_order)
    pct_all = (counts_all / counts_all.sum() * 100).round(1)
    pct_pacap = (counts_pacap / counts_pacap.sum() * 100).round(1) if counts_pacap.sum() > 0 else pd.Series([0.0] * 5, index=stage_order)

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, counts, pct, title_suffix in [
        (ax1, counts_all, pct_all, f'NCR Patients\n(n={counts_all.sum():,})'),
        (ax2, counts_pacap, pct_pacap, f'PACAP Subset\n(n={counts_pacap.sum():,})'),
    ]:
        bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
        ax.set_title(f'Stage Distribution - {title_suffix}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        for bar, count, pct_val in zip(bars, counts.values, pct.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{int(count)}\n({pct_val:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    max_count = max(counts_all.max(), counts_pacap.max())
    ax1.set_ylim(0, max_count * 1.15)
    ax2.set_ylim(0, max_count * 1.15)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'stage_distribution_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stage distribution plot to {plot_path}")
    plt.close()

def plot_resectability_distribution(clinical_paths=None, pacap_paths=None, output_dir='outputs/analysis_plots'):
    try:
        CM_COL = getattr(main_config, 'CM_COL', 'cm')
        ARTERIEEL_COL = getattr(main_config, 'ARTERIEEL_COL', 'arterieel_vaatbetr')
        VENEUS_COL = getattr(main_config, 'VENEUS_COL', 'veneus_vaatbetr')
    except Exception:
        CM_COL, ARTERIEEL_COL, VENEUS_COL = 'cm', 'arterieel_vaatbetr', 'veneus_vaatbetr'

    if clinical_paths is None:
        clinical_paths = ['data/dev_set.csv', 'data/test_set.csv']
    if pacap_paths is None:
        pacap_paths = ['data/dev_set_pacap.csv', 'data/test_set_pacap.csv']

    dfs_clin = [pd.read_csv(p, low_memory=False) for p in clinical_paths if os.path.exists(p)]
    dfs_pacap = [pd.read_csv(p, low_memory=False) for p in pacap_paths if os.path.exists(p)]

    if not dfs_clin or not dfs_pacap:
        print("Warning: Missing files. Skipping resectability plot.")
        return

    df_clinical = pd.concat(dfs_clin, ignore_index=True)
    if ID_COL in df_clinical.columns:
        df_clinical = df_clinical.drop_duplicates(subset=[ID_COL], keep='first')

    # Try to load update file for arterieel column
    update_path = 'data/24-00328_update.csv'
    if os.path.exists(update_path):
        try:
            df_update = pd.read_csv(update_path, sep=';', low_memory=False)
        except Exception:
            df_update = pd.read_csv(update_path, low_memory=False)
        if ID_COL in df_update.columns and ARTERIEEL_COL in df_update.columns:
            df_clinical = df_clinical.merge(df_update[[ID_COL, ARTERIEEL_COL]], on=ID_COL, how='left', suffixes=('', '_update'))
            if f'{ARTERIEEL_COL}_update' in df_clinical.columns:
                df_clinical[ARTERIEEL_COL] = df_clinical[ARTERIEEL_COL].fillna(df_clinical[f'{ARTERIEEL_COL}_update'])
                df_clinical = df_clinical.drop(columns=[f'{ARTERIEEL_COL}_update'])

    df_pacap = pd.concat(dfs_pacap, ignore_index=True)
    if ID_COL in df_pacap.columns:
        df_pacap = df_pacap.drop_duplicates(subset=[ID_COL], keep='first')

    print(f"\n{'=' * 80}\nPLOTTING RESECTABILITY STATUS DISTRIBUTION\n{'=' * 80}\n")

    required = [CM_COL, ARTERIEEL_COL, VENEUS_COL]
    missing = [c for c in required if c not in df_clinical.columns]
    if missing:
        print(f"Error: Required columns not found: {missing}. Skipping.")
        return

    pacap_ids = set(df_pacap[ID_COL].unique()) if ID_COL in df_pacap.columns else set()
    df_ncr_pacap = df_clinical[df_clinical[ID_COL].isin(pacap_ids)].copy() if ID_COL in df_clinical.columns else pd.DataFrame()

    def classify(row):
        if row[CM_COL] == 1:
            return "Metastatic"
        a, v = row[ARTERIEEL_COL], row[VENEUS_COL]
        if pd.isna(a) or a in [7, 8, 9] or pd.isna(v) or v in [7, 8, 9]:
            return "Missing Data"
        if a == 0 and v in [0, 1, 2]:
            return "Resectable"
        if (a in [1, 2] and v in [0, 1, 2, 3]) or (a == 0 and v == 3):
            return "Borderline"
        if a in [3, 4] or v == 4:
            return "LAPC"
        return "Unclassified"

    df_clinical['resectability_status'] = df_clinical.apply(classify, axis=1)
    if len(df_ncr_pacap) > 0:
        df_ncr_pacap['resectability_status'] = df_ncr_pacap.apply(classify, axis=1)

    status_order = ['Resectable', 'Borderline', 'LAPC', 'Metastatic', 'Missing Data', 'Unclassified']
    counts_all = df_clinical['resectability_status'].value_counts().reindex(status_order, fill_value=0)
    counts_pacap = df_ncr_pacap['resectability_status'].value_counts().reindex(status_order, fill_value=0) if len(df_ncr_pacap) > 0 else pd.Series([0] * 6, index=status_order)
    pct_all = (counts_all / counts_all.sum() * 100).round(1)
    pct_pacap = (counts_pacap / counts_pacap.sum() * 100).round(1) if counts_pacap.sum() > 0 else pd.Series([0.0] * 6, index=status_order)

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6', '#9b59b6']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, counts, pct, title_suffix in [
        (ax1, counts_all, pct_all, f'NCR Patients\n(n={counts_all.sum():,})'),
        (ax2, counts_pacap, pct_pacap, f'PACAP Subset\n(n={counts_pacap.sum():,})'),
    ]:
        bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Resectability Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
        ax.set_title(f'Resectability Status Distribution - {title_suffix}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        for bar, count, pct_val in zip(bars, counts.values, pct.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{int(count)}\n({pct_val:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    max_count = max(counts_all.max(), counts_pacap.max())
    ax1.set_ylim(0, max_count * 1.15)
    ax2.set_ylim(0, max_count * 1.15)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'resectability_distribution_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved resectability distribution plot to {plot_path}")
    plt.close()

def plot_treatment_timing_distributions():
    try:
        DIAGNOSIS_DATE_COL = getattr(main_config, 'DIAGNOSIS_DATE_COL', 'dx_datum')
    except Exception:
        DIAGNOSIS_DATE_COL = 'dx_datum'

    print(f"\n{'=' * 80}\nPLOTTING TREATMENT TIMING DISTRIBUTIONS\n{'=' * 80}\n")

    test_clinical_path = 'data/black_envelop_test_set.csv' if os.path.exists('data/black_envelop_test_set.csv') else 'data/test_set.csv'
    test_pacap_path = 'data/black_envelop_test_set_pacap.csv' if os.path.exists('data/black_envelop_test_set_pacap.csv') else 'data/test_set_pacap.csv'

    if not os.path.exists(test_clinical_path) or not os.path.exists(test_pacap_path):
        print("Error: Test data not found. Skipping.")
        return

    df_clin = pd.read_csv(test_clinical_path, low_memory=False)
    df_pacap = pd.read_csv(test_pacap_path, low_memory=False)
    pacap_ids = set(df_pacap[ID_COL].unique())
    df_clin = df_clin[df_clin[ID_COL].isin(pacap_ids)].copy()

    if DIAGNOSIS_DATE_COL in df_pacap.columns:
        df_merged = pd.merge(df_clin, df_pacap[[ID_COL, DIAGNOSIS_DATE_COL]].drop_duplicates(ID_COL), on=ID_COL, how='left')
    else:
        df_merged = df_clin.copy()
        df_merged[DIAGNOSIS_DATE_COL] = np.nan

    if 'chir_int1' in df_merged.columns:
        df_merged['surgery_time_months'] = pd.to_numeric(df_merged['chir_int1'], errors='coerce') / 30.44
    else:
        df_merged['surgery_time_months'] = np.nan

    df_merged['has_surgery'] = (pd.to_numeric(df_merged.get('chir', 0), errors='coerce') == 1)

    syst_cols = [c for c in df_merged.columns if str(c).startswith('syst_start_int')]
    if syst_cols:
        def first_syst(row):
            for col in syst_cols:
                v = pd.to_numeric(row[col], errors='coerce')
                if pd.notna(v):
                    return v
            return np.nan
        df_merged['first_syst_time_months'] = df_merged.apply(first_syst, axis=1) / 30.44
        df_merged['has_chemo'] = df_merged['first_syst_time_months'].notna()
    else:
        df_merged['first_syst_time_months'] = np.nan
        df_merged['has_chemo'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    surgery_times = df_merged[df_merged['has_surgery']]['surgery_time_months'].dropna()
    chemo_times = df_merged[df_merged['has_chemo']]['first_syst_time_months'].dropna()

    for ax, times, title, color in [
        (axes[0, 0], surgery_times, f'Surgery Timing (n={len(surgery_times)})', '#3498db'),
        (axes[0, 1], chemo_times, f'First Systemic Treatment Timing (n={len(chemo_times)})', '#e74c3c'),
    ]:
        if len(times) > 0:
            ax.hist(times, bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(times.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {times.median():.1f}m')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Time Since Diagnosis (months)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    for ax, col, has_col, labels, title in [
        (axes[1, 0], 'has_surgery', df_merged.get('has_surgery'), ['No Surgery', 'Surgery'], 'Surgery Status'),
        (axes[1, 1], 'has_chemo', df_merged.get('has_chemo'), ['No Chemo', 'Chemo'], 'Chemotherapy Status'),
    ]:
        counts = df_merged[col].value_counts()
        vals = [counts.get(False, 0), counts.get(True, 0)]
        ax.bar(labels, vals, color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black')
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n(Total: {len(df_merged)} patients)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs/analysis_plots", exist_ok=True)
    plot_path = 'outputs/analysis_plots/treatment_timing_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved treatment timing plots to {plot_path}")
    plt.close()

def _analyze_rmse_by_treatment_status(treatment_type='surgery'):
    from xai_decision_making.models.synthesis.condor_core import ConDORConfig, OrdinalRegression, RDM, TDM
    from xai_decision_making.experiments.qol_data_ablation_study import (
        StaticEncoderBundle,
        build_sequences_from_merged,
        evaluate_sequences,
        load_merged,
    )

    treatment_label = "SURGERY" if treatment_type == 'surgery' else "CHEMOTHERAPY"
    print(f"\n{'=' * 80}\nANALYZING FULL RMSE BY {treatment_label} STATUS\n{'=' * 80}\n")

    test_clinical_path = 'data/black_envelop_test_set.csv'
    test_pacap_path = 'data/black_envelop_test_set_pacap.csv'

    if not os.path.exists(test_clinical_path) or not os.path.exists(test_pacap_path):
        print("Error: Black envelope test set not found!")
        return

    df_pacap_raw = pd.read_csv(test_pacap_path, low_memory=False)
    df_clin_raw = pd.read_csv(test_clinical_path, low_memory=False)
    pacap_ids = set(df_pacap_raw[ID_COL].unique())
    df_clin_raw = df_clin_raw[df_clin_raw[ID_COL].isin(pacap_ids)].copy()
    print(f"Filtered to PACAP patients: {len(df_clin_raw)}")

    if treatment_type == 'surgery':
        if 'chir' not in df_clin_raw.columns:
            print("Warning: 'chir' column not found. Skipping.")
            return
        df_clin_raw['chir'] = pd.to_numeric(df_clin_raw['chir'], errors='coerce')
        df_clin_raw['has_treatment'] = df_clin_raw['chir'] == 1
    else:
        syst_cols = [c for c in df_clin_raw.columns if str(c).startswith('syst_start_int')]
        if not syst_cols:
            print("Warning: No syst_start_int columns found. Skipping.")
            return
        df_clin_raw['has_treatment'] = df_clin_raw[syst_cols].notna().any(axis=1)

    patients_with = set(df_clin_raw[df_clin_raw['has_treatment']][ID_COL].astype(str).unique())
    patients_without = set(df_clin_raw[~df_clin_raw['has_treatment']][ID_COL].astype(str).unique())
    print(f"With {treatment_type}: {len(patients_with)} | Without: {len(patients_without)}")

    results_by_study = []

    for study_name, folder_name in STUDY_MAPPING.items():
        study_dir = f"outputs/{folder_name}"
        if not os.path.exists(study_dir):
            continue

        all_ckpts = sorted(glob.glob(f"{study_dir}/fold_*_rmse_*.pt"))
        if not all_ckpts:
            print(f"No checkpoints for {study_name}")
            continue

        # Group by fold, take most recent per fold
        fold_groups = defaultdict(list)
        for fpath in all_ckpts:
            m = re.search(r'fold_(\d+)_rmse_', os.path.basename(fpath))
            if m:
                fold_groups[int(m.group(1))].append(fpath)

        checkpoint_files = []
        for fold_num in sorted(fold_groups.keys()):
            if fold_num <= 10:
                best = sorted(fold_groups[fold_num], key=lambda p: os.path.getmtime(p), reverse=True)[0]
                checkpoint_files.append(best)

        print(f"\nProcessing {study_name} ({len(checkpoint_files)} folds)...")
        fold_results = []

        for fold_idx, ckpt_path in enumerate(checkpoint_files):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                config_args = ckpt['config_args']
                enc_dict = ckpt.get('encoders', {})

                conf = ConDORConfig(**config_args)

                ord_net = OrdinalRegression(conf).to(conf.device)
                rdm_net = RDM(conf).to(conf.device)
                tdm_net = TDM(conf).to(conf.device)

                if 'ordinal' in ckpt and ckpt['ordinal'] is not None:
                    ord_net.load_state_dict(ckpt['ordinal'])
                rdm_net.load_state_dict(ckpt['rdm'])
                tdm_net.load_state_dict(ckpt['tdm'])
                ord_net.eval(); rdm_net.eval(); tdm_net.eval()

                # Reconstruct encoders using the same pattern as evaluate_condor.py
                enc = None
                use_static_any = config_args.get('use_static_rdm', False) or config_args.get('use_static_tdm', False)
                if use_static_any and enc_dict.get('static_num_imputer') is not None:
                    enc = StaticEncoderBundle(
                        num_imputer=enc_dict['static_num_imputer'],
                        num_scaler=enc_dict['static_num_scaler'],
                        cat_imputer=enc_dict['static_cat_imputer'],
                        cat_encoder=enc_dict['static_cat_encoder'],
                        num_cols=enc_dict['static_num_cols'],
                        cat_cols=enc_dict['static_cat_cols'],
                    )

                df_merged = load_merged(test_pacap_path, test_clinical_path)
                all_seqs, _, _ = build_sequences_from_merged(
                    df_merged,
                    df_clin_raw,
                    use_static=use_static_any,
                    use_treatment=config_args.get('use_treatment', False),
                    encoders=enc,
                    fit_encoders=(enc is None and use_static_any),
                    risk_lookup=None,
                )

                seqs_with = [s for s in all_seqs if str(s.get('patient_id', '')) in patients_with]
                seqs_without = [s for s in all_seqs if str(s.get('patient_id', '')) in patients_without]
                print(f"  Fold {fold_idx + 1}: {len(seqs_with)} with / {len(seqs_without)} without {treatment_type}")

                models = (ord_net, rdm_net, tdm_net)

                def eval_group(seqs):
                    if not seqs:
                        return {'Full_RMSE': float('nan'), 'RDM_RMSE': float('nan')}
                    return evaluate_sequences(
                        models=models,
                        config=conf,
                        sequences=seqs,
                        use_static_rdm=config_args.get('use_static_rdm', False),
                        use_static_tdm=config_args.get('use_static_tdm', False),
                        use_treatment=config_args.get('use_treatment', False),
                        use_mortality_risk=config_args.get('use_mortality_risk', False),
                        use_ordinal=False,
                    )

                cap = treatment_label.capitalize()
                m_with = eval_group(seqs_with)
                m_without = eval_group(seqs_without)

                fold_row = {'fold': fold_idx + 1}
                for key in m_with:
                    fold_row[f'{key}_With_{cap}'] = m_with[key]
                    fold_row[f'{key}_No_{cap}'] = m_without.get(key, float('nan'))
                fold_results.append(fold_row)

                print(f"    Full RMSE: With={m_with.get('Full_RMSE', float('nan')):.3f} | Without={m_without.get('Full_RMSE', float('nan')):.3f}")

            except Exception as e:
                import traceback
                print(f"  Error fold {fold_idx + 1}: {e}")
                traceback.print_exc()

        if fold_results:
            df_fold = pd.DataFrame(fold_results)
            row = {'Study': study_name}
            cap = treatment_label.capitalize()
            for col in df_fold.columns:
                if col == 'fold':
                    continue
                row[f'{col}_Mean'] = df_fold[col].mean(skipna=True)
                row[f'{col}_Std'] = df_fold[col].std(skipna=True)
            results_by_study.append(row)

    if results_by_study:
        df_out = pd.DataFrame(results_by_study)
        os.makedirs("outputs/analysis_tables", exist_ok=True)
        csv_path = f'outputs/analysis_tables/rmse_by_{treatment_type}_status.csv'
        df_out.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results to {csv_path}")
    else:
        print("No results to analyze.")


def analyze_rmse_by_surgery_status():
    _analyze_rmse_by_treatment_status(treatment_type='surgery')


def analyze_rmse_by_chemo_status():
    _analyze_rmse_by_treatment_status(treatment_type='chemo')

def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--eval-set', type=str, default='both',
                        choices=['Black_Envelope', 'CV_Validation', 'both', 'all'])
    parser.add_argument('--skip-data-plots', action='store_true',
                        help='Skip the QoL/stage/resectability distribution plots')
    parser.add_argument('--analyze-by-treatment', action='store_true',
                        help='Run RMSE-by-treatment-status analysis (loads model checkpoints, slow)')
    args = parser.parse_args()

    if not args.skip_data_plots:
        dev_pacap_path = 'data/dev_set_pacap.csv'
        test_pacap_path = 'data/test_set_pacap.csv'
        if os.path.exists(dev_pacap_path) and os.path.exists(test_pacap_path):
            combined = pd.concat([
                pd.read_csv(dev_pacap_path, low_memory=False),
                pd.read_csv(test_pacap_path, low_memory=False),
            ], ignore_index=True)
            temp_path = 'data/dev_test_combined_pacap_temp.csv'
            combined.to_csv(temp_path, index=False)
            plot_qol_responses_per_month(temp_path)
            plot_responses_per_patient(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        plot_qol_responses_per_month('data/black_envelop_test_set_pacap.csv')
        plot_responses_per_patient('data/black_envelop_test_set_pacap.csv')

        plot_stage_distribution(
            clinical_paths=['data/dev_set.csv', 'data/test_set.csv'],
            pacap_paths=['data/dev_set_pacap.csv', 'data/test_set_pacap.csv'],
        )
        plot_resectability_distribution(
            clinical_paths=['data/dev_set.csv', 'data/test_set.csv'],
            pacap_paths=['data/dev_set_pacap.csv', 'data/test_set_pacap.csv'],
        )
        plot_treatment_timing_distributions()

    if args.analyze_by_treatment:
        analyze_rmse_by_surgery_status()
        analyze_rmse_by_chemo_status()

    eval_sets = ['CV_Validation', 'Black_Envelope'] if args.eval_set == 'both' else [args.eval_set]

    for eval_set in eval_sets:
        label = eval_set.replace('_', ' ')
        print(f"\n{'=' * 80}\n{label.upper()}\n{'=' * 80}")
        results = load_all_results(eval_set=eval_set)
        if not results:
            print(f"No {eval_set} results found.")
            continue

        rdm_metrics, full_metrics = perform_statistical_tests(results, eval_set_name=label)
        create_summary_table(results, eval_set_name=label)
        plot_summary(results, eval_set_name=label, rdm_metrics=rdm_metrics, full_metrics=full_metrics)
        compare_baseline_vs_rdm_tdm(results, eval_set_name=label)

        if eval_set == 'Black_Envelope':
            all_df = pd.concat([df.assign(Study=name) for name, df in results.items()], ignore_index=True)
            print(f"\n{'=' * 80}\nBLACK ENVELOPE SUMMARY STATISTICS\n{'=' * 80}")
            summary_cols = [c for c in ['Full_RMSE', 'RDM_RMSE', 'RMSE_1M', 'RMSE_3M', 'RMSE_6M', 'RMSE_9M', 'RMSE_12M'] if c in all_df.columns]
            for col in summary_cols:
                print(f"\n{col}:")
                for study in sorted(all_df['Study'].unique()):
                    sub = all_df[all_df['Study'] == study][col]
                    print(f"  {study}: {sub.mean():.3f} ± {sub.std():.3f} (n={len(sub)} folds)")


if __name__ == "__main__":
    main()