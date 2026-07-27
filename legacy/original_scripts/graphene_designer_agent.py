import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
from scipy.spatial import cKDTree
import datetime # For unique run folder

# --- Configuration ---
PDB_PATH = "/home/akshay/Documents/MD/test_gemini/material_RL/topo/4040.pdb"
LABELS_CSV = "/home/akshay/Documents/MD/test_gemini/material_RL/labels/processed/noisy_train.csv"
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"

# Generate a unique run folder based on timestamp
RUN_DIR = os.path.join("/home/akshay/Documents/MD/test_gemini/gemini", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
IMG_DIR = os.path.join(RUN_DIR, "images")
PDB_OUTPUT_DIR = os.path.join(RUN_DIR, "pdbs")

MIN_REJECTION = 0.60
MAX_REMOVALS = 30 # Reverted to 30 steps
TRIALS_PER_STEP = 10 # Reverted to 10 trials per step
N_ATOMS_PER_TRIAL = 3 # Reverted to max 3 atoms to remove per trial

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(PDB_OUTPUT_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MLP Definition (from LLMAGENT_cleaned.py) ---
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

# --- Load normalization stats ---
train_df = pd.read_csv(LABELS_CSV)
flux_mean, flux_std = train_df['flux'].mean(), train_df['flux'].std()
rej_mean, rej_std = train_df['rejection'].mean(), train_df['rejection'].std()

# --- Load Models ---
def load_model(model, ckpt):
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get('state_dict', sd)
    model.load_state_dict(sd)
    return model.to(device).eval()

resnet_flux = load_model(models.resnet50(pretrained=True), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_flux.ckpt"))
mlp_flux = load_model(MLP(in_size=1000), os.path.join(MODEL_DIR, "mlp_smoothl1_all2_flux.ckpt"))
resnet_rej  = load_model(models.resnet50(pretrained=True), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_rej.ckpt"))
mlp_rej     = load_model(MLP(in_size=1000), os.path.join(MODEL_DIR, "mlp_smoothl1_all2_rej.ckpt"))

# --- Prediction Functions ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_flux_rejection(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        flux_features = resnet_flux(batch_t)
        rej_features = resnet_rej(batch_t)
        
        water_flux_standardized, _ = mlp_flux(flux_features.view(flux_features.size(0), -1))
        water_flux = (water_flux_standardized.item() * flux_std) + flux_mean

        ion_rejection_standardized, _ = mlp_rej(rej_features.view(rej_features.size(0), -1))
        ion_rejection = (ion_rejection_standardized.item() * rej_std) + rej_mean
        ion_rejection = max(0.0, min(1.0, ion_rejection))

    return water_flux, ion_rejection

# --- PDB Parsing and Graphene Representation ---
xy_coords = None # This will be a numpy array of shape (num_atoms, 2)
original_pdb_lines = [] # Store original PDB lines to reconstruct

def parse_pdb(pdb_path):
    global xy_coords, original_pdb_lines
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            original_pdb_lines.append(line)
            if line.startswith("ATOM") and line[13:15] == "C ": # Only consider Carbon atoms
                x = float(line[30:38])
                y = float(line[38:46])
                coords.append([x, y])
    xy_coords = np.array(coords)
    print(f"Loaded {len(xy_coords)} carbon atoms from PDB.")

# --- PDB Saving ---
def save_pdb(removed_atom_indices, step_label):
    pdb_path = os.path.join(PDB_OUTPUT_DIR, f"step_{step_label}.pdb")
    with open(pdb_path, 'w') as f:
        for i, line in enumerate(original_pdb_lines):
            if line.startswith("ATOM") and line[13:15] == "C ":
                # PDB atom index is 1-based, so adjust for 0-based numpy array
                atom_idx_in_array = int(line[6:11]) - 1 
                if atom_idx_in_array not in removed_atom_indices:
                    f.write(line)
            else:
                f.write(line)
    return pdb_path

# --- Image Generation ---
def generate_mock_image(atom_ids, label):
    path = os.path.join(IMG_DIR, f"step_{label}.png")
    plt.figure(figsize=(5, 5)) # Set a fixed figure size for consistent image generation
    plt.scatter(xy_coords[:, 0], xy_coords[:, 1], s=6, c='black')

    # Highlight removed atoms in white
    if len(atom_ids) > 0:
        plt.scatter(xy_coords[atom_ids, 0], xy_coords[atom_ids, 1], s=30, c='white')
    
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0) # Save with tight bounding box
    plt.close()
    return path

# --- Main Design Loop ---
def design_graphene_nanopore():
    parse_pdb(PDB_PATH)
    
    # Create a KDTree for efficient nearest neighbor search
    atom_tree = cKDTree(xy_coords)

    # Initial state (Step 0)
    removed_atoms = set()
    current_flux = 0.0
    current_rejection = 1.0 # Initial rejection is 100%
    
    # Log results
    design_log = []
    
    # Save initial PDB and image
    initial_image_path = generate_mock_image(list(removed_atoms), 0)
    initial_pdb_path = save_pdb(list(removed_atoms), 0)

    design_log.append({
        "step": 0,
        "removed_atoms_count": len(removed_atoms),
        "flux": current_flux,
        "rejection": current_rejection,
        "image_path": initial_image_path,
        "pdb_path": initial_pdb_path
    })
    print(f"Step 0: Flux={current_flux:.2f}, Rejection={current_rejection:.2f}")

    # Design loop
    for step in range(1, MAX_REMOVALS + 1):
        best_trial_flux = -1.0
        best_trial_rejection = -1.0
        best_trial_removed_atoms = set()
        best_trial_image_path = ""
        best_trial_pdb_path = ""

        candidate_atoms_for_removal = set()
        if len(removed_atoms) == 0:
            # If no atoms removed yet, find the central atom
            center_x = np.mean(xy_coords[:, 0])
            center_y = np.mean(xy_coords[:, 1])
            distances = np.linalg.norm(xy_coords - np.array([center_x, center_y]), axis=1)
            central_atom_idx = np.argmin(distances)
            candidate_atoms_for_removal.add(central_atom_idx)
        else:
            # Find neighbors of removed atoms (Ozark-like colonization)
            for atom_id in removed_atoms:
                # Query for 30 closest neighbors, excluding already removed atoms
                distances, indices = atom_tree.query(xy_coords[atom_id], k=31) # k+1 to exclude self
                for i, dist in zip(indices, distances):
                    if i != atom_id and i not in removed_atoms:
                        candidate_atoms_for_removal.add(i)
        
        # Convert to list for random sampling
        candidate_atoms_list = list(candidate_atoms_for_removal)

        if not candidate_atoms_list and len(removed_atoms) > 0:
            print(f"No more candidate atoms to remove at step {step}. Stopping.")
            break
        elif not candidate_atoms_list and len(removed_atoms) == 0:
            print(f"No central atom found. Exiting.")
            break

        trials_data = [] # To store data for all trials in this step

        for trial in range(TRIALS_PER_STEP):
            trial_removed_atoms = set(removed_atoms)
            
            # Randomly select 1 to N_ATOMS_PER_TRIAL from candidates
            num_to_remove = random.randint(1, N_ATOMS_PER_TRIAL)
            
            # Ensure we don't try to remove more atoms than available candidates
            if len(candidate_atoms_list) < num_to_remove:
                num_to_remove = len(candidate_atoms_list)
            
            new_atoms_to_remove = []
            if num_to_remove > 0:
                new_atoms_to_remove = random.sample(candidate_atoms_list, num_to_remove)
                trial_removed_atoms.update(new_atoms_to_remove)
            
            if len(trial_removed_atoms) > MAX_REMOVALS * N_ATOMS_PER_TRIAL * 2: # A rough upper bound to prevent infinite loops
                # print(f"Trial {trial} at step {step} exceeded MAX_REMOVALS. Skipping.")
                continue

            # Generate image for this trial
            trial_image_label = f"{step}_{trial}"
            trial_image_path = generate_mock_image(list(trial_removed_atoms), trial_image_label)

            # Predict flux and rejection
            trial_flux, trial_rejection = predict_flux_rejection(trial_image_path)

            trials_data.append({
                "trial_num": trial,
                "new_atoms_removed": new_atoms_to_remove,
                "total_removed_atoms": len(trial_removed_atoms),
                "flux": trial_flux,
                "rejection": trial_rejection,
                "image_path": trial_image_path
            })

            # Selection criteria: higher flux, maintaining rejection
            if trial_rejection >= MIN_REJECTION:
                if trial_flux > best_trial_flux:
                    best_trial_flux = trial_flux
                    best_trial_rejection = trial_rejection
                    best_trial_removed_atoms = trial_removed_atoms
                    best_trial_image_path = trial_image_path
                elif trial_flux == best_trial_flux and trial_rejection > best_trial_rejection:
                    best_trial_flux = trial_flux
                    best_trial_rejection = trial_rejection
                    best_trial_removed_atoms = trial_removed_atoms
                    best_trial_image_path = trial_image_path
        
        if not best_trial_removed_atoms: # If no valid trial was found (e.g., all failed rejection criteria)
            print(f"No valid trials found for step {step} that meet rejection criteria. Stopping.")
            break

        removed_atoms = best_trial_removed_atoms
        current_flux = best_trial_flux
        current_rejection = best_trial_rejection

        # Save PDB for the best trial of this step
        best_trial_pdb_path = save_pdb(list(removed_atoms), step)

        design_log.append({
            "step": step,
            "removed_atoms_count": len(removed_atoms),
            "flux": current_flux,
            "rejection": current_rejection,
            "image_path": best_trial_image_path,
            "pdb_path": best_trial_pdb_path
        })
        print(f"Step {step}: Removed Atoms={len(removed_atoms)}, Flux={current_flux:.2f}, Rejection={current_rejection:.2f}")

    # Save design log to CSV
    log_df = pd.DataFrame(design_log)
    log_df.to_csv(os.path.join(RUN_DIR, "graphene_design_log.csv"), index=False)
    print(f"Design log saved to {os.path.join(RUN_DIR, 'graphene_design_log.csv')}")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Flux vs Removed Atoms
    plt.subplot(1, 2, 1)
    plt.plot(log_df['removed_atoms_count'], log_df['flux'], marker='o', linestyle='-')
    plt.xlabel('Removed Atoms Count')
    plt.ylabel('Water Flux')
    plt.title('Water Flux vs. Removed Atoms')
    plt.grid(True)

    # Rejection vs Removed Atoms
    plt.subplot(1, 2, 2)
    plt.plot(log_df['removed_atoms_count'], log_df['rejection'], marker='o', linestyle='-')
    plt.xlabel('Removed Atoms Count')
    plt.ylabel('Ion Rejection')
    plt.title('Ion Rejection vs. Removed Atoms')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "flux_rejection_vs_removed_atoms.png"))
    plt.close()
    print(f"Plots saved to {os.path.join(RUN_DIR, 'flux_rejection_vs_removed_atoms.png')}")

    # Flux and Rejection vs Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Water Flux', color=color)
    ax1.plot(log_df['step'], log_df['flux'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Ion Rejection', color=color)
    ax2.plot(log_df['step'], log_df['rejection'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Water Flux and Ion Rejection vs. Steps')
    plt.savefig(os.path.join(RUN_DIR, "flux_rejection_vs_steps.png"))
    plt.close()
    print(f"Combined plot saved to {os.path.join(RUN_DIR, 'flux_rejection_vs_steps.png')}")

if __name__ == "__main__":
    design_graphene_nanopore()