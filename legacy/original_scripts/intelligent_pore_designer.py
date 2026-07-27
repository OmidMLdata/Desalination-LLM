import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cdist
import shutil
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial import ConvexHull, cKDTree, QhullError
import imageio

# --- Configuration ---
PDB_PATH = "/home/akshay/Documents/MD/test_gemini/material_RL/topo/4040.pdb"
LABELS_CSV = "/home/akshay/Documents/MD/test_gemini/material_RL/labels/processed/noisy_train.csv"
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"
IMG_DIR = "/home/akshay/Documents/MD/test_gemini/gemini"
GIF_PATH = os.path.join(IMG_DIR, "pore_evolution.gif")
MAX_REJECTION = 1.0
MIN_REJECTION = 0.60
MAX_REMOVALS = 90
TRIALS_PER_STEP = 10
N_ATOMS_PER_TRIAL = 3
NUM_STEPS = 50
TARGET_FLUX_START = 0
TARGET_FLUX_END = 350

# Ensure IMG_DIR exists
os.makedirs(IMG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model, path):
    sd = torch.load(path, map_location=device)
    sd = sd.get('state_dict', sd)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()

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

def load_pdb(pdb_path):
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atoms.append(line)
    return atoms

def write_pdb(atoms_lines, output_path):
    with open(output_path, 'w') as f:
        for line in atoms_lines:
            f.write(line)
    print(f"PDB written to {output_path}")

def pdb_to_image(atoms_obj, img_path, removed_indices):
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    positions = atoms_obj.get_positions()
    xy_coords = positions[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(xy_coords[:, 0], xy_coords[:, 1], s=6, c='black')

    if removed_indices:
        plt.scatter(xy_coords[list(removed_indices), 0], xy_coords[list(removed_indices), 1], s=30, c='red')
    
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(img_path, dpi=100)
    plt.close()
    return img_path

def load_mlp_model(model_path, in_size):
    model = MLP(in_size)
    sd = torch.load(model_path, map_location=device)
    # Handle potential 'state_dict' key
    if 'state_dict' in sd:
        sd = sd['state_dict']
    # Filter out unexpected keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in sd.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model.to(device)

transform = transforms.Compose([
    transforms.CenterCrop((350, 350)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_flux_rejection(image_path, resnet_flux, mlp_flux, resnet_rej, mlp_rej, flux_mean, flux_std_val, rej_mean, rej_std_val):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        resnet_out_flux = resnet_flux(x)
        flux_pred_std, _ = mlp_flux(resnet_out_flux)
        flux = float(flux_pred_std.item() * flux_std_val + flux_mean)

        resnet_out_rej = resnet_rej(x)
        rej_pred_std, _ = mlp_rej(resnet_out_rej)
        rejection = float(rej_pred_std.item() * rej_std_val + rej_mean)
        
    return flux, rejection

def get_pore_properties(atoms_obj, removed_indices):
    if len(removed_indices) < 3:
        return 0, 0

    positions = atoms_obj.get_positions()
    removed_positions = positions[list(removed_indices)]

    try:
        hull = ConvexHull(removed_positions[:, :2])
        area = hull.volume
        perimeter = hull.area
    except QhullError:
        area = 0
        perimeter = 0

    return area, perimeter

class GraphenePoreDesigner:
    def __init__(self):
        self.atoms_obj = read(PDB_PATH)
        self.original_atoms_lines = load_pdb(PDB_PATH)
        self.removed_atom_indices = set()
        self.best_flux_rejection_per_step = []
        self.trials_log = []

        # Load models
        self.resnet_flux = load_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_flux.ckpt"))
        self.resnet_rej = load_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_rej.ckpt"))
        
        # Get ResNet output size
        self.resnet_flux.fc = nn.Identity()
        self.resnet_rej.fc = nn.Identity()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        resnet_output_size = self.resnet_flux(dummy_input).shape[1]
        
        self.mlp_flux = load_mlp_model(os.path.join(MODEL_DIR, "mlp_smoothl1_all2_flux.ckpt"), in_size=resnet_output_size)
        self.mlp_rej = load_mlp_model(os.path.join(MODEL_DIR, "mlp_smoothl1_all2_rej.ckpt"), in_size=resnet_output_size)

        # Load standardization parameters from LABELS_CSV
        train_df = pd.read_csv(LABELS_CSV)
        self.flux_mean, self.flux_std = train_df['flux'].mean(), train_df['flux'].std()
        self.rej_mean, self.rej_std = train_df['rejection'].mean(), train_df['rejection'].std()

        # Initialize cKDTree for atom proximity search
        self.xy_coords = self.atoms_obj.get_positions()[:, :2]
        self.tree = cKDTree(self.xy_coords)
        
        # Identify edge atoms to prevent their removal
        hull = ConvexHull(self.xy_coords)
        self.edge_atom_indices = set(hull.vertices)

    def get_neighbors(self, current_removed_atoms, k=30):
        unremoved_indices = [i for i in range(len(self.atoms_obj)) if i not in current_removed_atoms and i not in self.edge_atom_indices]
        
        if not unremoved_indices:
            return []

        if not current_removed_atoms:
            center = np.mean(self.xy_coords, axis=0)
            distances = cdist([center], self.xy_coords[unremoved_indices])[0]
            sorted_indices_in_unremoved = np.argsort(distances)
            return [unremoved_indices[i] for i in sorted_indices_in_unremoved[:k]]
        else:
            removed_coords = self.xy_coords[list(current_removed_atoms)]
            distances = cdist(self.xy_coords[unremoved_indices], removed_coords).min(axis=1)
            sorted_unremoved_indices = np.argsort(distances)
            return [unremoved_indices[i] for i in sorted_unremoved_indices[:k]]

    def run_design(self, num_steps=NUM_STEPS):
        image_sequence = []
        target_flux_schedule = np.linspace(TARGET_FLUX_START, TARGET_FLUX_END, num_steps + 1)

        # Step 0: Initial state
        initial_img_path = os.path.join(IMG_DIR, f"step_0.png")
        pdb_to_image(self.atoms_obj, initial_img_path, self.removed_atom_indices)
        image_sequence.append(initial_img_path)
        
        self.best_flux_rejection_per_step.append({
            "step": 0, "flux": 0.0, "rejection": 1.0, "removed_atoms_count": 0,
            "pore_area": 0, "pore_perimeter": 0
        })
        print(f"Step 0: Flux=0.0, Rejection=1.0, Removed Atoms=0")

        for step in range(1, num_steps + 1):
            print(f"--- Running Step {step} ---")
            best_trial_info = {'flux_diff': float('inf')}
            target_flux = target_flux_schedule[step]

            candidate_atoms = self.get_neighbors(self.removed_atom_indices, k=30)
            if not candidate_atoms:
                print("No more candidate atoms to remove. Stopping.")
                break

            for trial in range(TRIALS_PER_STEP):
                temp_removed_indices = set(self.removed_atom_indices)
                num_atoms_to_remove = np.random.randint(0, N_ATOMS_PER_TRIAL + 1)
                
                if num_atoms_to_remove > 0 and len(candidate_atoms) > 0:
                    num_atoms_to_remove = min(num_atoms_to_remove, len(candidate_atoms))
                    newly_removed_candidates = np.random.choice(
                        [idx for idx in candidate_atoms if idx not in temp_removed_indices],
                        size=num_atoms_to_remove, replace=False
                    ).tolist()
                    temp_removed_indices.update(newly_removed_candidates)

                if len(temp_removed_indices) > MAX_REMOVALS: continue

                trial_img_path = os.path.join(IMG_DIR, f"step_{step}_trial_{trial}.png")
                pdb_to_image(self.atoms_obj, trial_img_path, temp_removed_indices)

                predicted_flux, predicted_rejection = predict_flux_rejection(
                    trial_img_path, self.resnet_flux, self.mlp_flux, self.resnet_rej, self.mlp_rej,
                    self.flux_mean, self.flux_std, self.rej_mean, self.rej_std
                )
                
                flux_diff = abs(predicted_flux - target_flux)

                self.trials_log.append({
                    "step": step, "trial": trial, "flux": predicted_flux, "rejection": predicted_rejection,
                    "removed_atoms_count": len(temp_removed_indices), "flux_diff": flux_diff
                })

                if flux_diff < best_trial_info['flux_diff'] and predicted_rejection >= MIN_REJECTION:
                    best_trial_info = {
                        'flux': predicted_flux, 'rejection': predicted_rejection,
                        'removed_indices': temp_removed_indices, 'img_path': trial_img_path,
                        'flux_diff': flux_diff
                    }

            if 'removed_indices' in best_trial_info:
                self.removed_atom_indices = best_trial_info['removed_indices']
                step_img_path = os.path.join(IMG_DIR, f"step_{step}.png")
                shutil.copy(best_trial_info['img_path'], step_img_path)
                image_sequence.append(step_img_path)

                pore_area, pore_perimeter = get_pore_properties(self.atoms_obj, self.removed_atom_indices)
                
                self.best_flux_rejection_per_step.append({
                    "step": step, "flux": best_trial_info['flux'], "rejection": best_trial_info['rejection'],
                    "removed_atoms_count": len(self.removed_atom_indices),
                    "pore_area": pore_area, "pore_perimeter": pore_perimeter
                })
                print(f"Step {step} Result: Flux={best_trial_info['flux']:.2f}, Rejection={best_trial_info['rejection']:.2f}, Removed Atoms={len(self.removed_atom_indices)}")
            else:
                print(f"Step {step}: No suitable trial found. Continuing with previous best state.")
                self.best_flux_rejection_per_step.append(self.best_flux_rejection_per_step[-1])

        # Save results
        pd.DataFrame(self.best_flux_rejection_per_step).to_csv(os.path.join(IMG_DIR, "graphene_design_aggregated_results.csv"), index=False)
        pd.DataFrame(self.trials_log).to_csv(os.path.join(IMG_DIR, "graphene_design_trials_log.csv"), index=False)
        
        # Save final PDB
        final_atoms_lines = [line for i, line in enumerate(self.original_atoms_lines) if i not in self.removed_atom_indices]
        write_pdb(final_atoms_lines, os.path.join(IMG_DIR, "final_pore_structure.pdb"))

        # Generate GIF
        with imageio.get_writer(GIF_PATH, mode='I', duration=0.5) as writer:
            for path in image_sequence:
                writer.append_data(imageio.imread(path))
        print(f"Animation saved to {GIF_PATH}")

        # Plot results
        results_df = pd.DataFrame(self.best_flux_rejection_per_step)
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Water Flux', color=color)
        ax1.plot(results_df["step"], results_df["flux"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Ion Rejection', color=color)
        ax2.plot(results_df["step"], results_df["rejection"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "flux_rejection_plot.png"))
        print(f"Plot saved to {os.path.join(IMG_DIR, 'flux_rejection_plot.png')}")

if __name__ == "__main__":
    designer = GraphenePoreDesigner()
    designer.run_design()