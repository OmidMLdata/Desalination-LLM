import os
import pandas as pd
import matplotlib.pyplot as plt
from ozark_inspired_designer import GraphenePoreDesigner

IMG_DIR_BASE = "/home/akshay/Documents/MD/test_gemini/gemini/Atom_Removal/geminit20r3k30"

def generate_run_plots(results_df, run_dir):
    # Plot 1: Water Flux vs. Removed Atoms
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['removed_atoms_count'], results_df['flux'], marker='o')
    plt.xlabel('Removed Atoms Count')
    plt.ylabel('Water Flux')
    plt.title('Water Flux vs. Removed Atoms')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "water_flux_vs_removed_atoms.png"))
    plt.close()

    # Plot 2: Ion Rejection vs. Removed Atoms
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['removed_atoms_count'], results_df['rejection'], marker='o')
    plt.xlabel('Removed Atoms Count')
    plt.ylabel('Ion Rejection')
    plt.title('Ion Rejection vs. Removed Atoms')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "ion_rejection_vs_removed_atoms.png"))
    plt.close()

    # Plot 3: Combined Flux and Rejection vs. Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Water Flux', color=color)
    ax1.plot(results_df["step"], results_df["flux"], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Ion Rejection', color=color)
    ax2.plot(results_df["step"], results_df["rejection"], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Water Flux and Ion Rejection vs. Steps')
    plt.savefig(os.path.join(run_dir, "flux_rejection_vs_steps.png"))
    plt.close()

def generate_overlaid_plots(combined_results_df):
    # Plot 1: Water Flux vs. Removed Atoms (All Runs)
    plt.figure(figsize=(10, 6))
    for run_num, group in combined_results_df.groupby('run'):
        plt.plot(group['removed_atoms_count'], group['flux'], label=f'Run {run_num}', alpha=0.7, marker='o')
    plt.xlabel('Removed Atoms Count')
    plt.ylabel('Water Flux')
    plt.title('Water Flux vs. Removed Atoms (All Runs)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR_BASE, "overlaid_water_flux_vs_removed_atoms.png"))
    plt.close()

    # Plot 2: Ion Rejection vs. Removed Atoms (All Runs)
    plt.figure(figsize=(10, 6))
    for run_num, group in combined_results_df.groupby('run'):
        plt.plot(group['removed_atoms_count'], group['rejection'], label=f'Run {run_num}', alpha=0.7, marker='o')
    plt.xlabel('Removed Atoms Count')
    plt.ylabel('Ion Rejection')
    plt.title('Ion Rejection vs. Removed Atoms (All Runs)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR_BASE, "overlaid_ion_rejection_vs_removed_atoms.png"))
    plt.close()

    # Plot 3: Combined Flux and Rejection vs. Steps (All Runs)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    for run_num, group in combined_results_df.groupby('run'):
        ax1.plot(group["step"], group["flux"], alpha=0.7, label=f'Run {run_num} Flux', marker='o')
        ax2.plot(group["step"], group["rejection"], alpha=0.7, marker='x', label=f'Run {run_num} Rejection')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Water Flux')
    ax2.set_ylabel('Ion Rejection')
    ax1.grid(True)
    fig.legend()
    plt.title('Water Flux and Ion Rejection vs. Steps (All Runs)')
    plt.savefig(os.path.join(IMG_DIR_BASE, "overlaid_flux_rejection_vs_steps.png"))
    plt.close()

def run_multiple_simulations(num_runs=5):
    all_results = []
    all_trials = []

    for i in range(num_runs):
        print(f"--- Starting Run {i+1}/{num_runs} ---")
        run_dir = os.path.join(IMG_DIR_BASE, f"geminir1_{i+1}")
        designer = GraphenePoreDesigner(run_number=i)
        results_df, all_trials_df = designer.run_design(num_steps=30)
        
        results_df['run'] = i # Add run number to results
        all_trials_df['run'] = i # Add run number to trials

        os.makedirs(run_dir, exist_ok=True) # Create the run directory if it doesn't exist

        # Save individual run results
        results_df.to_csv(os.path.join(run_dir, "intelligent_design_results.csv"), index=False)
        all_trials_df.to_csv(os.path.join(run_dir, "all_trials_log.csv"), index=False)
        generate_run_plots(results_df, run_dir)

        all_results.append(results_df)
        all_trials.append(all_trials_df)

    # Concatenate all results and trials
    combined_results_df = pd.concat(all_results, ignore_index=True)
    combined_trials_df = pd.concat(all_trials, ignore_index=True)

    # Save combined trials to a single CSV
    combined_trials_df.to_csv(os.path.join(IMG_DIR_BASE, "all_runs_trials_log.csv"), index=False)

    # --- Data Aggregation and Analysis ---
    summary_df = combined_results_df.groupby('step').agg(
        flux_mean=('flux', 'mean'),
        flux_std=('flux', 'std'),
        rejection_mean=('rejection', 'mean'),
        rejection_std=('rejection', 'std'),
        removed_atoms_count_mean=('removed_atoms_count', 'mean')
    ).reset_index()

    # Save the summary report
    summary_report_path = os.path.join(IMG_DIR_BASE, "design_summary_report.csv")
    summary_df.to_csv(summary_report_path, index=False)
    print(f"Summary report saved to {summary_report_path}")

    # --- Plotting ---
    generate_overlaid_plots(combined_results_df)
    # Plot 4: Filtered Water vs. Step (0 to 50 removed atoms)
    filtered_water_df = combined_results_df.copy()
    filtered_water_df['filtered_water'] = filtered_water_df['flux'] * filtered_water_df['rejection']

    # Group by step and calculate mean and std for filtered water
    filtered_water_summary_df = filtered_water_df.groupby('step').agg(
        filtered_water_mean=('filtered_water', 'mean'),
        filtered_water_std=('filtered_water', 'std'),
        removed_atoms_count_mean=('removed_atoms_count', 'mean')
    ).reset_index()

    # Filter for removed atoms count between 0 and 50
    filtered_water_summary_df = filtered_water_summary_df[
        (filtered_water_summary_df['removed_atoms_count_mean'] >= 0) &
        (filtered_water_summary_df['removed_atoms_count_mean'] <= 50)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_water_summary_df['step'], filtered_water_summary_df['filtered_water_mean'], marker='o', linestyle='-')
    plt.fill_between(
        filtered_water_summary_df['step'],
        filtered_water_summary_df['filtered_water_mean'] - filtered_water_summary_df['filtered_water_std'],
        filtered_water_summary_df['filtered_water_mean'] + filtered_water_summary_df['filtered_water_std'],
        alpha=0.2
    )
    plt.xlabel('Step')
    plt.ylabel('Filtered Water (Mean)')
    plt.title('Filtered Water vs. Step (Removed Atoms 0-50)')
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR_BASE, "filtered_water_vs_step.png"))
    plt.close()

    print(f"All plots saved to {IMG_DIR_BASE}")

if __name__ == "__main__":
    run_multiple_simulations(num_runs=5)