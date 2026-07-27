import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import imageio
from scipy.spatial import cKDTree, ConvexHull

# Define constants
PDB_PATH = "/home/akshay/Documents/MD/test_gemini/material_RL/topo/4040.pdb"
LABELS_CSV = "/home/akshay/Documents/MD/test_gemini/material_RL/labels/processed/noisy_train.csv"
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"
BASE_IMG_DIR = "/home/akshay/Documents/MD/test_gemini/Removal/geminit10r2k30_runs" # Base directory for this specific run

MAX_REJECTION = 1.0
MIN_REJECTION = 0.60
MAX_REMOVALS = 60 # 30 steps * 2 atoms/step
TRIALS_PER_STEP = 10
N_ATOMS_PER_TRIAL_MAX = 2 # Maximum number of atoms to remove per trial
NUM_SIMULATION_RUNS = 5
NEIGHBORHOOD_SIZE = 30 # Number of closest neighbors to consider

# Ensure base image directory exists
os.makedirs(BASE_IMG_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse PDB file to get atom coordinates
def parse_pdb(pdb_path):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

xy_coords = parse_pdb(PDB_PATH)[:, :2] # Only need x, y for 2D representation

# Load normalization stats
train_df = pd.read_csv(LABELS_CSV)
flux_mean, flux_std = train_df['flux'].mean(), train_df['flux'].std()
rej_mean, rej_std = train_df['rejection'].mean(), train_df['rejection'].std()

# Define MLP class (from cnn.py)
class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim, bias=False),
        )

    def forward(self, x):
        return x + self.main(x)


class MLP(nn.Module):
    def __init__(self, in_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        out = self.model(x)
        return self.output(out), out

# Load models
def load_model(model, ckpt):
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get('state_dict', sd)
    model.load_state_dict(sd)
    return model.to(device).eval()

resnet_flux = load_model(models.resnet50(pretrained=True), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_flux.ckpt"))
mlp_flux = load_model(MLP(in_size=1000), os.path.join(MODEL_DIR, "mlp_smoothl1_all2_flux.ckpt"))
resnet_rej  = load_model(models.resnet50(pretrained=True), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_rej.ckpt"))
mlp_rej     = load_model(MLP(in_size=1000), os.path.join(MODEL_DIR, "mlp_smoothl1_all2_rej.ckpt"))

# Define transform
transform = transforms.Compose([
    transforms.CenterCrop((350, 350)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction functions
def flux_predict(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f_std, _ = mlp_flux(resnet_flux(x))
    return float(f_std.item() * flux_std + flux_mean)

def rej_predict(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        r_std, _ = mlp_rej(resnet_rej(x))
    return float(r_std.item() * rej_std + rej_mean)

def predict_flux_rejection(image_path):
    img = Image.open(image_path).convert("RGB")
    return {"flux": flux_predict(img), "rejection": rej_predict(img)}

# Image generation function
def generate_mock_image(atom_ids, label, current_img_dir, trial_label=None):
    if trial_label is not None:
        path = os.path.join(current_img_dir, f"step_{label}_trial_{trial_label}.png")
    else:
        path = os.path.join(current_img_dir, f"step_{label}.png")
    plt.figure()
    plt.scatter(xy_coords[:, 0], xy_coords[:, 1], s=6, c='black')

    # Ensure atom_ids is a numpy array of integers
    atom_ids = np.array(atom_ids, dtype=int)

    # Filter out atom_ids that are out of bounds
    valid_atom_ids = atom_ids[atom_ids < len(xy_coords)]

    if len(valid_atom_ids) > 0:
        plt.scatter(xy_coords[valid_atom_ids, 0], xy_coords[valid_atom_ids, 1], s=30, c='white')

    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, dpi=100)
    plt.close()
    return path

# Function to calculate pore area and perimeter using ConvexHull
def calculate_pore_properties(removed_atom_indices):
    if len(removed_atom_indices) < 3:
        return 0.0, 0.0 # Cannot form a polygon with less than 3 points
    
    # Get coordinates of the *remaining* atoms to define the pore boundary
    remaining_atom_indices = np.setdiff1d(np.arange(len(xy_coords)), removed_atom_indices)
    
    if len(remaining_atom_indices) < 3:
        return 0.0, 0.0 # Not enough remaining atoms to form a hull

    try:
        # Calculate convex hull of the *remaining* atoms to define the pore boundary
        hull = ConvexHull(xy_coords[remaining_atom_indices])
        return hull.area, hull.perimeter
    except Exception as e:
        print(f"Could not calculate ConvexHull for pore properties: {e}")
        return 0.0, 0.0

# Function to save PDB file with removed atoms
def save_pdb(removed_atom_indices, original_pdb_path, output_pdb_path):
    original_lines = []
    with open(original_pdb_path, 'r') as f:
        original_lines = f.readlines()

    with open(output_pdb_path, 'w') as f:
        atom_counter = 0
        for line in original_lines:
            if line.startswith("ATOM"):
                if atom_counter not in removed_atom_indices:
                    f.write(line)
                atom_counter += 1
            else:
                f.write(line) # Write non-ATOM lines (HEADER, CRYST1, END, etc.)

def run_simulation(run_idx, base_img_dir, max_removals, trials_per_step, n_atoms_per_trial_max, neighborhood_size):
    current_img_dir = os.path.join(base_img_dir, f"geminit10r2k30_{run_idx + 1}")
    os.makedirs(current_img_dir, exist_ok=True)

    LOG_FILE_RUN = os.path.join(current_img_dir, "graphene_design_log.csv")
    TRIAL_LOG_FILE_RUN = os.path.join(current_img_dir, "graphene_design_trials_log.csv")
    GIF_PATH_RUN = os.path.join(current_img_dir, "single_atom_chain.gif")

    current_removed_atoms = np.array([], dtype=int) # Stores indices of removed atoms
    all_removed_atoms_history = [] # To store removed atoms for GIF
    log_data = []
    trial_log_data = [] # New list to store all trial data

    # Step 0: Initial state (no atoms removed)
    step_0_image_path = generate_mock_image(current_removed_atoms, 0, current_img_dir)
    step_0_predictions = {"flux": 0.0, "rejection": 1.0} # Modified for flux=0, rejection=100

    best_flux_step = step_0_predictions['flux']
    best_rejection_step = step_0_predictions['rejection']
    
    # Calculate pore properties for step 0 (no atoms removed, so area/perimeter of the full sheet)
    # For a pristine sheet, pore area is 0, perimeter is 0 (no pore)
    pore_area_step_0 = 0.0
    pore_perimeter_step_0 = 0.0

    log_entry = {
        "step": 0,
        "removed_atoms_count": len(current_removed_atoms),
        "flux": best_flux_step,
        "rejection": best_rejection_step,
        "pore_area": pore_area_step_0,
        "pore_perimeter": pore_perimeter_step_0,
        "removed_atoms_indices": []
    }
    log_data.append(log_entry)
    all_removed_atoms_history.append(current_removed_atoms.copy())
    
    # Save PDB for step 0 (original sheet)
    save_pdb(current_removed_atoms, PDB_PATH, os.path.join(current_img_dir, f"step_0_pore.pdb"))

    print(f"Step 0: Initial Flux = {best_flux_step:.2f}, Rejection = {best_rejection_step:.2f}")

    # Find the center of the graphene sheet for initial atom removal
    center_x = (np.max(xy_coords[:, 0]) + np.min(xy_coords[:, 0])) / 2
    center_y = (np.max(xy_coords[:, 1]) + np.min(xy_coords[:, 1])) / 2
    center_point = np.array([center_x, center_y])

    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(xy_coords)

    # Main loop for 30 steps
    for step in range(1, 31):
        best_flux_trial = -1.0
        best_rejection_trial = -1.0
        best_trial_removed_atoms = current_removed_atoms.copy()
        best_pore_area_trial = 0.0
        best_pore_perimeter_trial = 0.0

        # Determine candidate atoms for removal in this step
        if len(current_removed_atoms) == 0:
            # If no atoms removed yet, select the NEIGHBORHOOD_SIZE closest atoms to the center
            distances_from_center = np.linalg.norm(xy_coords - center_point, axis=1)
            candidate_atoms_for_this_step = np.argsort(distances_from_center)[:neighborhood_size]
        else:
            # Find atoms closest to the currently removed atoms
            # Query k-nearest neighbors for each removed atom
            distances, indices = tree.query(xy_coords[current_removed_atoms], k=neighborhood_size + len(current_removed_atoms))
            candidate_atoms_for_this_step = np.unique(indices.flatten())
            
            # Exclude already removed atoms from candidates
            candidate_atoms_for_this_step = np.setdiff1d(candidate_atoms_for_this_step, current_removed_atoms)
            
            # Limit to NEIGHBORHOOD_SIZE closest available atoms if more than NEIGHBORHOOD_SIZE are found
            if len(candidate_atoms_for_this_step) > neighborhood_size:
                # Sort by distance to the closest removed atom
                if len(current_removed_atoms) > 0:
                    min_distances_to_removed = np.min(tree.query(xy_coords[candidate_atoms_for_this_step], k=1, workers=-1)[0][:, np.newaxis] + \
                                                      np.linalg.norm(xy_coords[candidate_atoms_for_this_step, np.newaxis, :] - \
                                                                     xy_coords[current_removed_atoms, :], axis=2), axis=1)
                    sorted_indices = np.argsort(min_distances_to_removed)
                    candidate_atoms_for_this_step = candidate_atoms_for_this_step[sorted_indices[:neighborhood_size]]
                else:
                    # Fallback if current_removed_atoms is somehow empty but candidates are many
                    candidate_atoms_for_this_step = candidate_atoms_for_this_step[:neighborhood_size]

        for trial in range(trials_per_step):
            # Randomly decide how many atoms to remove in this trial (0, 1, or 2)
            num_to_remove_this_trial = np.random.randint(0, n_atoms_per_trial_max + 1)

            # Ensure we don't try to remove more atoms than available candidates
            num_to_remove_this_trial = min(num_to_remove_this_trial, len(candidate_atoms_for_this_step))
            
            if num_to_remove_this_trial == 0:
                atoms_to_remove_this_trial = np.array([], dtype=int)
            else:
                atoms_to_remove_this_trial = np.random.choice(candidate_atoms_for_this_step, num_to_remove_this_trial, replace=False)

            trial_removed_atoms = np.unique(np.concatenate((current_removed_atoms, atoms_to_remove_this_trial))).astype(int)

            if len(trial_removed_atoms) > max_removals:
                continue # Skip if too many atoms are removed

            trial_image_path = generate_mock_image(trial_removed_atoms, step, current_img_dir, trial)
            trial_predictions = predict_flux_rejection(trial_image_path)
            trial_flux = trial_predictions['flux']
            trial_rejection = trial_predictions['rejection']
            
            # Calculate pore area and perimeter for the trial
            pore_area, pore_perimeter = calculate_pore_properties(trial_removed_atoms)

            # Log data for each trial
            trial_log_entry = {
                "step": step,
                "trial": trial,
                "removed_atoms_count": len(trial_removed_atoms),
                "flux": trial_flux,
                "rejection": trial_rejection,
                "pore_area": pore_area,
                "pore_perimeter": pore_perimeter,
                "removed_atoms_indices": trial_removed_atoms.tolist()
            }
            trial_log_data.append(trial_log_entry)

            # Evaluate trial for best in step
            if trial_rejection >= MIN_REJECTION:
                if trial_flux > best_flux_trial:
                    best_flux_trial = trial_flux
                    best_rejection_trial = trial_rejection
                    best_trial_removed_atoms = trial_removed_atoms.copy()
                    best_pore_area_trial = pore_area
                    best_pore_perimeter_trial = pore_perimeter

        # Update for the current step
        current_removed_atoms = best_trial_removed_atoms.copy()
        all_removed_atoms_history.append(current_removed_atoms.copy())

        log_entry = {
            "step": step,
            "removed_atoms_count": len(current_removed_atoms),
            "flux": best_flux_trial,
            "rejection": best_rejection_trial,
            "pore_area": best_pore_area_trial,
            "pore_perimeter": best_pore_perimeter_trial,
            "removed_atoms_indices": current_removed_atoms.tolist()
        }
        log_data.append(log_entry)
        
        # Save PDB for the best pore of this step
        save_pdb(current_removed_atoms, PDB_PATH, os.path.join(current_img_dir, f"step_{step}_pore.pdb"))

        print(f"Step {step}: Removed Atoms = {len(current_removed_atoms)}, Flux = {best_flux_trial:.2f}, Rejection = {best_rejection_trial:.2f}")

    # Save logs for this run
    pd.DataFrame(log_data).to_csv(LOG_FILE_RUN, index=False)
    pd.DataFrame(trial_log_data).to_csv(TRIAL_LOG_FILE_RUN, index=False)
    print(f"Run {run_idx + 1} logs saved to {current_img_dir}")

    # Generate GIF (requires imageio and Pillow)
    try:
        images = []
        for i, removed_atoms in enumerate(all_removed_atoms_history):
            image_path = generate_mock_image(removed_atoms, i, current_img_dir) # Pass current_img_dir
            images.append(imageio.imread(image_path))
        imageio.mimsave(GIF_PATH_RUN, images, fps=2)
        print(f"Run {run_idx + 1} GIF saved to {GIF_PATH_RUN}")
    except Exception as e:
        print(f"Error generating GIF for run {run_idx + 1}: {e}")

    return pd.DataFrame(log_data), pd.DataFrame(trial_log_data)

if __name__ == "__main__":
    configurations = {
        "gemini_runs": {
            "max_removals": 90,
            "trials_per_step": 10,
            "n_atoms_per_trial_max": 3,
            "neighborhood_size": 30
        },
        "geminir5_runs": {
            "max_removals": 150,
            "trials_per_step": 10,
            "n_atoms_per_trial_max": 5,
            "neighborhood_size": 30
        },
        "geminir2_runs": {
            "max_removals": 60,
            "trials_per_step": 10,
            "n_atoms_per_trial_max": 2,
            "neighborhood_size": 30
        },
        "geminir2k15_runs": {
            "max_removals": 60,
            "trials_per_step": 10,
            "n_atoms_per_trial_max": 2,
            "neighborhood_size": 15
        },
        "geminit20r2k45_runs": {
            "max_removals": 60,
            "trials_per_step": 20,
            "n_atoms_per_trial_max": 2,
            "neighborhood_size": 45
        },
        "geminit10r2k30_runs": {
            "max_removals": 60,
            "trials_per_step": 10,
            "n_atoms_per_trial_max": 2,
            "neighborhood_size": 30
        }
    }

    for base_dir_name, config in configurations.items():
        current_base_img_dir = os.path.join("/home/akshay/Documents/MD/test_gemini/Removal", base_dir_name)
        os.makedirs(current_base_img_dir, exist_ok=True)

        all_runs_log_data = []
        all_runs_trial_log_data = []

        print(f"\n--- Processing configuration: {base_dir_name} ---")
        for run_idx in range(NUM_SIMULATION_RUNS):
            print(f"\n--- Running Simulation {run_idx + 1}/{NUM_SIMULATION_RUNS} for {base_dir_name} ---")
            step_log_df, trial_log_df = run_simulation(
                run_idx,
                current_base_img_dir,
                config["max_removals"],
                config["trials_per_step"],
                config["n_atoms_per_trial_max"],
                config["neighborhood_size"]
            )
            step_log_df['run'] = run_idx
            trial_log_df['run'] = run_idx
            all_runs_log_data.append(step_log_df)
            all_runs_trial_log_data.append(trial_log_df)

        # Concatenate all runs
        full_step_log_df = pd.concat(all_runs_log_data, ignore_index=True)
        full_trial_log_df = pd.concat(all_runs_trial_log_data, ignore_index=True)

        # Aggregate results for mean and standard deviation
        aggregated_df = full_step_log_df.groupby('step').agg({
            'removed_atoms_count': 'mean',
            'flux': ['mean', 'std'],
            'rejection': ['mean', 'std'],
            'pore_area': ['mean', 'std'],
            'pore_perimeter': ['mean', 'std']
        }).reset_index()
        aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
        aggregated_df.rename(columns={'removed_atoms_count_mean': 'removed_atoms_count'}, inplace=True)
        aggregated_df.to_csv(os.path.join(current_base_img_dir, "graphene_design_aggregated_results.csv"), index=False)
        print(f"Aggregated results saved to {current_base_img_dir}")

        # --- Plotting Script for Aggregated Results ---
        # Load the aggregated data
        df = pd.read_csv(os.path.join(current_base_img_dir, "graphene_design_aggregated_results.csv"))

        # Plot 1: Water Flux vs Removed Atoms with error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['removed_atoms_count'], df['flux_mean'], yerr=df['flux_std'], fmt='o', linestyle='-', color='blue', capsize=5)
        plt.xlabel('Number of Removed Atoms')
        plt.ylabel('Water Flux (Mean ± Std Dev)')
        plt.title(f'Water Flux vs. Number of Removed Atoms (Aggregated - {base_dir_name})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(current_base_img_dir, 'flux_vs_removed_atoms_aggregated.png'))
        plt.close()

        # Plot 2: Ion Rejection vs Removed Atoms with error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['removed_atoms_count'], df['rejection_mean'], yerr=df['rejection_std'], fmt='o', linestyle='-', color='green', capsize=5)
        plt.xlabel('Number of Removed Atoms')
        plt.ylabel('Ion Rejection (Mean ± Std Dev)')
        plt.title(f'Ion Rejection vs. Number of Removed Atoms (Aggregated - {base_dir_name})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(current_base_img_dir, 'rejection_vs_removed_atoms_aggregated.png'))
        plt.close()

        # Plot 3: Water Flux and Ion Rejection vs Steps with error bars
        fig, ax1 = plt.subplots(figsize=(12, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Water Flux (Mean ± Std Dev)', color=color)
        ax1.errorbar(df['step_'], df['flux_mean'], yerr=df['flux_std'], fmt='o', linestyle='-', color=color, label='Water Flux', capsize=5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:green'
        ax2.set_ylabel('Ion Rejection (Mean ± Std Dev)', color=color)  # we already handled the x-label with ax1
        ax2.errorbar(df['step_'], df['rejection_mean'], yerr=df['rejection_std'], fmt='x', linestyle='--', color=color, label='Ion Rejection', capsize=5)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'Water Flux and Ion Rejection vs. Steps (Aggregated - {base_dir_name})')
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.savefig(os.path.join(current_base_img_dir, 'flux_rejection_vs_steps_aggregated.png'))
        plt.close()

        print(f"Aggregated plots saved to {current_base_img_dir}")