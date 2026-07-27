import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

plt.rc("font", size=16,family='serif')

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"
IMG_DIR_BASE = "/home/akshay/Documents/MD/test_gemini/gemini/Atom_Removal/geminit20r3k30" # Corrected path

# --- Model Loading Functions ---
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

# --- Image Transformation ---
transform = transforms.Compose([
    transforms.CenterCrop((350, 350)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Feature Extraction Function ---
def extract_features(image_path, resnet_model, mlp_model):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        resnet_features = resnet_model(x) # Output of ResNet before MLP
        _, mlp_features = mlp_model(resnet_features) # Output of MLP's model part (64-dim)
    return mlp_features.cpu().numpy().flatten()

# --- Main Visualization Logic ---
def visualize_tsne(num_runs=5):
    # Load models
    resnet_flux = load_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_flux.ckpt"))
    resnet_rej = load_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), os.path.join(MODEL_DIR, "resnet_smoothl1_all2_rej.ckpt"))
    mlp_flux = load_mlp_model(os.path.join(MODEL_DIR, "mlp_smoothl1_all2_flux.ckpt"), in_size=1000)
    mlp_rej = load_mlp_model(os.path.join(MODEL_DIR, "mlp_smoothl1_all2_rej.ckpt"), in_size=1000)

    # --- Collect data from generated trials ---
    all_flux_features = []
    all_rej_features = []
    all_flux_values = []
    all_rej_values = []

    print("Collecting data from generated trials...")
    for i in range(num_runs):
        run_dir = os.path.join(IMG_DIR_BASE, f"geminir1_{i+1}") # Corrected run_dir construction
        all_trials_log_path = os.path.join(run_dir, "all_trials_log.csv")
        
        if not os.path.exists(all_trials_log_path):
            print(f"Warning: {all_trials_log_path} not found. Skipping run {i}.")
            continue

        trials_df = pd.read_csv(all_trials_log_path)
        for idx, row in tqdm(trials_df.iterrows(), total=len(trials_df), desc=f"Run {i} trials"):
            image_path = row['image_path']
            # The image_path in the CSV is relative to the run_dir, so we need to join it
            full_image_path = os.path.join(run_dir, os.path.basename(image_path)) # Corrected image path construction
            
            if os.path.exists(full_image_path):
                all_flux_features.append(extract_features(full_image_path, resnet_flux, mlp_flux))
                all_rej_features.append(extract_features(full_image_path, resnet_rej, mlp_rej))
                all_flux_values.append(row['flux'])
                all_rej_values.append(row['rejection'])
            else:
                print(f"Warning: Image not found at {full_image_path}. Skipping.")

    if not all_flux_features:
        print("No features to visualize. Exiting.")
        return

    # Convert lists to numpy arrays
    all_flux_features = np.array(all_flux_features)
    all_rej_features = np.array(all_rej_features)
    all_flux_values = np.array(all_flux_values)
    all_rej_values = np.array(all_rej_values)

    # --- Apply t-SNE ---
    print("Applying t-SNE for flux features...")
    tsne_flux = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    flux_embedding = tsne_flux.fit_transform(all_flux_features)

    print("Applying t-SNE for rejection features...")
    tsne_rej = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    rej_embedding = tsne_rej.fit_transform(all_rej_features)

    # --- Plotting ---
    # Plot for Water Flux
    plt.figure(figsize=(10, 6))
    scatter_flux = plt.scatter(
        flux_embedding[:, 0], flux_embedding[:, 1],
        c=all_flux_values, cmap='cividis', marker='o', s=50, alpha=0.4
    )

    plt.colorbar(scatter_flux, label=r'Water Flux(# $\mathregular{ns^{-1}})$')
    plt.title('2D t-SNE Embedding of Water Flux Features (Generated Trials)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(False)

    # Remove all plot frame spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR_BASE, "tsne_flux_embedding_colored.png"), dpi=600)
    plt.close()

    # Plot for Ion Rejection
    plt.figure(figsize=(10, 6))
    scatter_rej = plt.scatter(
        rej_embedding[:, 0], rej_embedding[:, 1],
        c=all_rej_values, cmap='cividis', marker='o', s=50, alpha=0.4
    )

    # Single colorbar with percentage formatting
    cbar = plt.colorbar(scatter_rej)
    formatter = FuncFormatter(lambda x, _: f'{x*100:.0f}%')  # Format ticks as percentages
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.set_ticks(np.linspace(0.0, 1.0, 11))
    cbar.set_label('Ion Rejection (%)', fontsize=16)

    # Plot labels and style
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)

    # Remove frame
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR_BASE, "tsne_rejection_embedding_colored.png"), dpi=600)
    plt.close()

    print(f"t-SNE plots saved to {IMG_DIR_BASE}")

if __name__ == "__main__":
    visualize_tsne(num_runs=5)
