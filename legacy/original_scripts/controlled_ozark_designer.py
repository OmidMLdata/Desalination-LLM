
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial import ConvexHull, cKDTree, QhullError
from scipy.spatial.distance import cdist
import imageio
import logging

# --- Configuration ---
PDB_PATH = "/home/akshay/Documents/MD/test_gemini/material_RL/topo/4040.pdb"
LABELS_CSV = "/home/akshay/Documents/MD/test_gemini/material_RL/labels/processed/noisy_train.csv"
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"
IMG_DIR = "/home/akshay/Documents/MD/test_gemini/gemini/controlled_ozark"
LOG_FILE = os.path.join(IMG_DIR, "graphene_design_log.csv")
TERMINAL_LOG_FILE = os.path.join(IMG_DIR, "terminal_log.txt")
NUM_RUNS = 5
NUM_STEPS = 30
TRIALS_PER_STEP = 10
MAX_REMOVALS_PER_STEP = 3
MIN_REJECTION = 0.6
TARGET_FLUX_MAX = 300

# --- Setup ---
os.makedirs(IMG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(TERMINAL_LOG_FILE, mode='w'),
                        logging.StreamHandler()
                    ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ControlledOzarkDesigner:
    def __init__(self, run_num):
        self.run_num = run_num
        self.atoms_obj = read(PDB_PATH)
        self.xy_coords = self.atoms_obj.get_positions()[:, :2]
        self.original_atoms_lines = self.load_pdb(PDB_PATH)
        self.removed_atom_indices = set()
        self.results = []
        self.trials_log = []
        self.run_dir = os.path.join(IMG_DIR, f"run_{self.run_num}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.load_models()
        self.load_scaling_params()

        hull = ConvexHull(self.xy_coords)
        self.edge_atom_indices = set(hull.vertices)
        self.tree = cKDTree(self.xy_coords)
        self.transform = transforms.Compose([
            transforms.CenterCrop((350, 350)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_pdb(self, pdb_path):
        with open(pdb_path, 'r') as f:
            return [line for line in f if line.startswith("ATOM")]

    def load_models(self):
        self.resnet_flux = self.load_resnet("resnet_smoothl1_all2_flux.ckpt")
        self.resnet_rej = self.load_resnet("resnet_smoothl1_all2_rej.ckpt")
        self.mlp_flux = self.load_mlp("mlp_smoothl1_all2_flux.ckpt", 1000)
        self.mlp_rej = self.load_mlp("mlp_smoothl1_all2_rej.ckpt", 1000)

    def load_resnet(self, model_name):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1000)
        model_path = os.path.join(MODEL_DIR, model_name)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd.get('state_dict', sd), strict=False)
        return model.to(device).eval()

    def load_mlp(self, model_name, in_size):
        model = nn.Sequential(nn.Linear(in_size, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        model_path = os.path.join(MODEL_DIR, model_name)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd.get('state_dict', sd), strict=False)
        return model.to(device).eval()

    def load_scaling_params(self):
        train_df = pd.read_csv(LABELS_CSV)
        self.flux_mean, self.flux_std = train_df['flux'].mean(), train_df['flux'].std()
        self.rej_mean, self.rej_std = train_df['rejection'].mean(), train_df['rejection'].std()

    def get_neighbors(self, k=30):
        unremoved_indices = [i for i in range(len(self.atoms_obj)) if i not in self.removed_atom_indices and i not in self.edge_atom_indices]
        if not self.removed_atom_indices:
            center = np.mean(self.xy_coords, axis=0)
            distances = cdist([center], self.xy_coords[unremoved_indices])[0]
            return [unremoved_indices[i] for i in np.argsort(distances)[:k]]
        else:
            removed_coords = self.xy_coords[list(self.removed_atom_indices)]
            distances = cdist(self.xy_coords[unremoved_indices], removed_coords).min(axis=1)
            return [unremoved_indices[i] for i in np.argsort(distances)[:k]]

    def run(self):
        logging.info(f"Starting Run {self.run_num}")
        target_flux_schedule = np.linspace(0, TARGET_FLUX_MAX, NUM_STEPS + 1)
        self.results.append({'step': 0, 'removed_atoms': 0, 'flux': 0, 'rejection': 1.0, 'run': self.run_num})

        for step in range(1, NUM_STEPS + 1):
            best_trial = {"flux_diff": float('inf')}
            target_flux = target_flux_schedule[step]
            candidate_atoms = self.get_neighbors()

            if not candidate_atoms:
                logging.info("No more atoms to remove.")
                break

            for trial in range(TRIALS_PER_STEP):
                temp_removed_indices = set(self.removed_atom_indices)
                num_to_remove = np.random.randint(1, MAX_REMOVALS_PER_STEP + 1)
                num_to_remove = min(num_to_remove, len(candidate_atoms))
                newly_removed = np.random.choice(candidate_atoms, num_to_remove, replace=False).tolist()
                temp_removed_indices.update(newly_removed)

                img_path, pdb_path = self.generate_mock_image(list(temp_removed_indices), f"step_{step}_trial_{trial}")
                predictions = self.predict_flux_rejection(img_path)
                flux_diff = abs(predictions['flux'] - target_flux)

                self.trials_log.append({
                    'run': self.run_num, 'step': step, 'trial': trial, 
                    'removed_atoms': len(temp_removed_indices), 'flux': predictions['flux'], 
                    'rejection': predictions['rejection'], 'flux_diff': flux_diff, 'pdb_path': pdb_path
                })

                if flux_diff < best_trial["flux_diff"] and predictions['rejection'] >= MIN_REJECTION:
                    best_trial = predictions
                    best_trial['removed_indices'] = temp_removed_indices
                    best_trial['flux_diff'] = flux_diff
            
            if 'removed_indices' in best_trial:
                self.removed_atom_indices = best_trial['removed_indices']
                self.results.append({'step': step, 'removed_atoms': len(self.removed_atom_indices), 'flux': best_trial['flux'], 'rejection': best_trial['rejection'], 'run': self.run_num})
                logging.info(f"Run {self.run_num}, Step {step}: Target Flux={target_flux:.2f}, Best Flux={best_trial['flux']:.2f}, Rejection={best_trial['rejection']:.2f}, Removed={len(self.removed_atom_indices)}")
            else:
                self.results.append(self.results[-1])
                logging.info(f"Run {self.run_num}, Step {step}: No improvement. Staying at {len(self.removed_atom_indices)} removed atoms.")

        return pd.DataFrame(self.results), pd.DataFrame(self.trials_log)

    def generate_mock_image(self, atom_ids, label):
        path = os.path.join(self.run_dir, f"{label}.png")
        pdb_path = os.path.join(self.run_dir, f"{label}.pdb")
        plt.figure()
        plt.scatter(self.xy_coords[:, 0], self.xy_coords[:, 1], s=6, c='black')
        if atom_ids:
            plt.scatter(self.xy_coords[atom_ids, 0], self.xy_coords[atom_ids, 1], s=30, c='white')
        plt.axis('off'); plt.gca().set_aspect('equal', adjustable='box'); plt.savefig(path, dpi=100); plt.close()
        self.write_pdb([i for i in range(len(self.original_atoms_lines)) if i not in self.removed_atom_indices.union(atom_ids)], pdb_path)
        return path, pdb_path

    def write_pdb(self, kept_atom_indices, output_path):
        with open(output_path, 'w') as f:
            for i in kept_atom_indices:
                f.write(self.original_atoms_lines[i])

    def predict_flux_rejection(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            flux_features = self.resnet_flux(x)
            rej_features = self.resnet_rej(x)
            f_std = self.mlp_flux(flux_features)
            r_std = self.mlp_rej(rej_features)
        flux = float(f_std.item() * self.flux_std + self.flux_mean)
        rejection = float(r_std.item() * self.rej_std + self.rej_mean)
        return {"flux": flux, "rejection": rejection}

def main():
    all_results = []
    all_trials = []
    for i in range(NUM_RUNS):
        agent = ControlledOzarkDesigner(run_num=i+1)
        results_df, trials_df = agent.run()
        all_results.append(results_df)
        all_trials.append(trials_df)

    final_results_df = pd.concat(all_results)
    final_trials_df = pd.concat(all_trials)
    final_trials_df.to_csv(LOG_FILE, index=False)

    summary = final_results_df.groupby('step').agg(
        mean_flux=('flux', 'mean'), std_flux=('flux', 'std'),
        mean_rejection=('rejection', 'mean'), std_rejection=('rejection', 'std'),
        mean_removed_atoms=('removed_atoms', 'mean'), std_removed_atoms=('removed_atoms', 'std')
    ).reset_index()
    summary.to_csv(os.path.join(IMG_DIR, "design_summary.csv"), index=False)
    logging.info("Design summary saved.")
    plot_results(summary)

def plot_results(summary):
    plt.style.use('seaborn-v0_8-whitegrid')
    # Flux vs Removed Atoms
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['mean_removed_atoms'], summary['mean_flux'], yerr=summary['std_flux'], fmt='-o', capsize=5, label='Mean Water Flux')
    plt.xlabel("Removed Atoms"); plt.ylabel("Water Flux"); plt.title("Water Flux vs. Removed Atoms"); plt.legend();
    plt.savefig(os.path.join(IMG_DIR, "flux_vs_removed_atoms.png")); plt.close()

    # Rejection vs Removed Atoms
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['mean_removed_atoms'], summary['mean_rejection'], yerr=summary['std_rejection'], fmt='-o', capsize=5, label='Mean Ion Rejection')
    plt.xlabel("Removed Atoms"); plt.ylabel("Ion Rejection"); plt.title("Ion Rejection vs. Removed Atoms"); plt.legend();
    plt.savefig(os.path.join(IMG_DIR, "rejection_vs_removed_atoms.png")); plt.close()

    # Combined Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Water Flux', color='tab:red')
    ax1.errorbar(summary['step'], summary['mean_flux'], yerr=summary['std_flux'], fmt='-o', color='tab:red', capsize=5, label='Mean Water Flux')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ion Rejection', color='tab:blue')
    ax2.errorbar(summary['step'], summary['mean_rejection'], yerr=summary['std_rejection'], fmt='-o', color='tab:blue', capsize=5, label='Mean Ion Rejection')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.title("Water Flux and Ion Rejection vs. Steps"); fig.tight_layout();
    plt.savefig(os.path.join(IMG_DIR, "flux_rejection_vs_steps.png")); plt.close()
    logging.info("Plots saved.")

if __name__ == "__main__":
    main()
