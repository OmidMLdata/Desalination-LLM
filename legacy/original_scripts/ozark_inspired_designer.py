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
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import cKDTree

# --- Configuration ---
PDB_PATH = "/home/akshay/Documents/MD/test_gemini/material_RL/topo/4040.pdb"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_CSV = "/home/akshay/Documents/MD/test_gemini/material_RL/labels/processed/noisy_train.csv"
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"
IMG_DIR_BASE = "/home/akshay/Documents/MD/test_gemini/gemini/Removal"
MAX_REJECTION = 1.0
MIN_REJECTION = 0.60
MAX_REMOVALS = 90 # This is the max number of atoms to remove, not steps
TRIALS_PER_STEP = 20
N_ATOMS_PER_TRIAL = 3 # Max atoms to remove per trial
MAX_ATOMS_TO_ADD = 3 # Max atoms to add per step

def load_model(model, path):
    sd = torch.load(path, map_location=device, weights_only=True)
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

def load_mlp_model(model_path, in_size):
    model = MLP(in_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    return model.to(device)

def load_pdb(pdb_path):
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_id = int(line[6:11])
                atoms.append({
                    "atom_id": atom_id,
                    "original_line": line
                })
    return pd.DataFrame(atoms)

def write_pdb(df_atoms, output_path):
    with open(output_path, 'w') as f:
        for _, row in df_atoms.iterrows():
            f.write(row["original_line"])

def pdb_to_image(atoms_obj, img_path, removed_indices):
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    positions = atoms_obj.get_positions()
    xy_coords = positions[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(xy_coords[:, 0], xy_coords[:, 1], s=6, c='black')

    if removed_indices:
        plt.scatter(xy_coords[list(removed_indices), 0], xy_coords[list(removed_indices), 1], s=30, c='white')
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(img_path, dpi=100)
    plt.close()
    return img_path

transform = transforms.Compose([
    transforms.CenterCrop((350, 350)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_flux_rejection(image_path, resnet_flux, mlp_flux, resnet_rej, mlp_rej, flux_mean, flux_std, rej_mean, rej_std):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f_std, _ = mlp_flux(resnet_flux(x))
        r_std, _ = mlp_rej(resnet_rej(x))
    flux = float(f_std.item() * flux_std + flux_mean)
    rejection = float(r_std.item() * rej_std + rej_mean)
    rejection = max(0.0, min(1.0, rejection))
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
    def __init__(self, run_number):
        self.run_number = run_number
        self.img_dir = os.path.join(IMG_DIR_BASE, f"geminir1_{self.run_number + 1}")
        os.makedirs(self.img_dir, exist_ok=True)
        self.gif_path = os.path.join(self.img_dir, "single_atom_chain.gif")

        self.atoms_obj = read(PDB_PATH)
        self.original_atoms_df = load_pdb(PDB_PATH)
        self.removed_atom_indices = set()

        self.resnet_flux = load_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_flux.ckpt"))
        self.resnet_rej = load_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_rej.ckpt"))

        self.mlp_flux = load_mlp_model(os.path.join(MODEL_DIR, "mlp_smoothl1_all2_flux.ckpt"), in_size=1000)
        self.mlp_rej = load_mlp_model(os.path.join(MODEL_DIR, "mlp_smoothl1_all2_rej.ckpt"), in_size=1000)

        train_df = pd.read_csv(LABELS_CSV)
        self.flux_mean, self.flux_std = train_df['flux'].mean(), train_df['flux'].std()
        self.rej_mean, self.rej_std = train_df['rejection'].mean(), train_df['rejection'].std()

        self.xy_coords = self.atoms_obj.get_positions()[:, :2]
        self.tree = cKDTree(self.xy_coords)

    def get_neighbors(self, current_removed_atoms, k=30):
        unremoved_indices = [i for i in range(len(self.atoms_obj)) if i not in current_removed_atoms]
        
        if not unremoved_indices:
            return []

        if not current_removed_atoms:
            center = np.mean(self.xy_coords, axis=0)
            distances = cdist([center], self.xy_coords[unremoved_indices])[0]
            sorted_indices_in_unremoved = np.argsort(distances)
            # Corrected line: Convert unremoved_indices to numpy array for advanced indexing
            return np.array(unremoved_indices)[sorted_indices_in_unremoved[:k]].tolist()
        else:
            removed_coords = self.xy_coords[list(current_removed_atoms)]
            distances = cdist(self.xy_coords[unremoved_indices], removed_coords).min(axis=1)
            sorted_unremoved_indices = np.argsort(distances)
            # Corrected line: Convert unremoved_indices to numpy array for advanced indexing
            return np.array(unremoved_indices)[sorted_unremoved_indices[:k]].tolist()

    def run_design(self, num_steps=30):
        results = []
        all_trials_log = []
        image_sequence = []

        initial_img_path = os.path.join(self.img_dir, f"step_0.png")
        pdb_to_image(self.atoms_obj, initial_img_path, self.removed_atom_indices)
        image_sequence.append(initial_img_path)
        
        initial_pdb_path = os.path.join(self.img_dir, f"step_0.pdb")
        write_pdb(self.original_atoms_df, initial_pdb_path)

        results.append({
            "step": 0,
            "flux": 0.0,
            "rejection": 1.0,
            "removed_atoms_count": 0,
            "pore_area": 0,
            "pore_perimeter": 0,
            "pdb_path": initial_pdb_path
        })

        # Add initial step 0 to all_trials_log
        all_trials_log.append({
            "run": self.run_number,
            "step": 0,
            "trial": 0,
            "removed_atoms_count": 0,
            "flux": 0.0,
            "rejection": 1.0,
            "image_path": initial_img_path,
            "pdb_path": initial_pdb_path
        })

        for step in tqdm(range(1, num_steps + 1), desc=f"Run {self.run_number}"):
            best_trial_flux = -1.0
            best_trial_rejection = -1.0
            best_trial_removed_indices = set()
            best_trial_img_path = None

            candidate_atoms = self.get_neighbors(self.removed_atom_indices, k=30)
            if not candidate_atoms:
                break

            for trial in range(TRIALS_PER_STEP):
                temp_removed_indices = set(self.removed_atom_indices)
                num_atoms_to_remove = np.random.randint(0, N_ATOMS_PER_TRIAL + 1)
                num_atoms_to_remove = min(num_atoms_to_remove, len(candidate_atoms))

                if num_atoms_to_remove > 0:
                    newly_removed_candidates = np.random.choice(
                        [idx for idx in candidate_atoms if idx not in temp_removed_indices],
                        size=num_atoms_to_remove,
                        replace=False
                    ).tolist()
                    temp_removed_indices.update(newly_removed_candidates)
                
                if len(temp_removed_indices) > MAX_REMOVALS:
                    continue

                trial_image_label = f"step_{step}_trial_{trial}.png"
                trial_img_path = os.path.join(self.img_dir, trial_image_label)
                pdb_to_image(self.atoms_obj, trial_img_path, temp_removed_indices)

                trial_pdb_path = os.path.join(self.img_dir, f"step_{step}_trial_{trial}.pdb")
                current_pdb_df = self.original_atoms_df[~self.original_atoms_df['atom_id'].isin([idx + 1 for idx in temp_removed_indices])]
                write_pdb(current_pdb_df, trial_pdb_path)

                predicted_flux, predicted_rejection = predict_flux_rejection(
                    trial_img_path, self.resnet_flux, self.mlp_flux, self.resnet_rej, self.mlp_rej,
                    self.flux_mean, self.flux_std, self.rej_mean, self.rej_std
                )

                all_trials_log.append({
                    "run": self.run_number,
                    "step": step,
                    "trial": trial,
                    "removed_atoms_count": len(temp_removed_indices),
                    "flux": predicted_flux,
                    "rejection": predicted_rejection,
                    "image_path": trial_img_path,
                    "pdb_path": trial_pdb_path
                })

                if (predicted_rejection >= MIN_REJECTION and predicted_flux > best_trial_flux) or \
                   (predicted_rejection < MIN_REJECTION and predicted_rejection > best_trial_rejection):
                    best_trial_flux = predicted_flux
                    best_trial_rejection = predicted_rejection
                    best_trial_removed_indices = temp_removed_indices
                    best_trial_img_path = trial_img_path
            
            newly_removed_atoms = best_trial_removed_indices - self.removed_atom_indices
            if len(newly_removed_atoms) > MAX_ATOMS_TO_ADD:
                final_new_atoms = set(np.random.choice(list(newly_removed_atoms), size=MAX_ATOMS_TO_ADD, replace=False))
                final_removed_for_step = self.removed_atom_indices.union(final_new_atoms)
            else:
                final_removed_for_step = set(best_trial_removed_indices)

            if len(final_removed_for_step) >= 3:
                removed_coords_for_hull = self.xy_coords[list(final_removed_for_step)]
                try:
                    hull = ConvexHull(removed_coords_for_hull)
                    all_atom_indices = np.arange(len(self.atoms_obj))
                    unremoved_atoms_indices = np.array([i for i in all_atom_indices if i not in final_removed_for_step])

                    if len(unremoved_atoms_indices) > 0:
                        unremoved_atoms_coords = self.xy_coords[unremoved_atoms_indices]
                        points_to_check = np.hstack((unremoved_atoms_coords, np.ones((len(unremoved_atoms_coords), 1))))
                        is_inside = np.all(np.dot(points_to_check, hull.equations.T) <= 1e-9, axis=1)
                        internal_unremoved_indices = unremoved_atoms_indices[is_inside]
                        
                        if len(internal_unremoved_indices) > 0:
                            num_already_added = len(final_removed_for_step) - len(self.removed_atom_indices)
                            num_to_add_from_internal = MAX_ATOMS_TO_ADD - num_already_added

                            if num_to_add_from_internal > 0:
                                if len(internal_unremoved_indices) > num_to_add_from_internal:
                                    internal_unremoved_indices = np.random.choice(internal_unremoved_indices, size=num_to_add_from_internal, replace=False)
                                final_removed_for_step.update(internal_unremoved_indices)

                            re_eval_image_label = f"step_{step}_colonized.png"
                            re_eval_img_path = os.path.join(self.img_dir, re_eval_image_label)
                            pdb_to_image(self.atoms_obj, re_eval_img_path, final_removed_for_step)
                            
                            re_eval_flux, re_eval_rejection = predict_flux_rejection(
                                re_eval_img_path, self.resnet_flux, self.mlp_flux, self.resnet_rej, self.mlp_rej,
                                self.flux_mean, self.flux_std, self.rej_mean, self.rej_std
                            )
                            
                            best_trial_flux = re_eval_flux
                            best_trial_rejection = re_eval_rejection
                            best_trial_removed_indices = final_removed_for_step
                            best_trial_img_path = re_eval_img_path
                            
                except QhullError:
                    pass

            if best_trial_img_path is not None and len(best_trial_removed_indices) > len(self.removed_atom_indices):
                self.removed_atom_indices = best_trial_removed_indices
                step_img_path = os.path.join(self.img_dir, f"step_{step}.png")
                shutil.copy(best_trial_img_path, step_img_path)
                image_sequence.append(step_img_path)

                current_pdb_df = self.original_atoms_df[~self.original_atoms_df['atom_id'].isin([idx + 1 for idx in self.removed_atom_indices])]
                step_pdb_path = os.path.join(self.img_dir, f"step_{step}.pdb")
                write_pdb(current_pdb_df, step_pdb_path)

                pore_area, pore_perimeter = get_pore_properties(self.atoms_obj, self.removed_atom_indices)

                results.append({
                    "run": self.run_number,
                    "step": step,
                    "flux": best_trial_flux,
                    "rejection": best_trial_rejection,
                    "removed_atoms_count": len(self.removed_atom_indices),
                    "pore_area": pore_area,
                    "pore_perimeter": pore_perimeter,
                    "pdb_path": step_pdb_path
                })
            else:
                results.append(results[-1])

        results_df = pd.DataFrame(results)
        all_trials_df = pd.DataFrame(all_trials_log)
        return results_df, all_trials_df

if __name__ == '__main__':
    designer = GraphenePoreDesigner(run_number=0)
    designer.run_design(num_steps=30)
