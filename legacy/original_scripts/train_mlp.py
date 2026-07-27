
import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
MODEL_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/models"
LABELS_CSV = "/home/akshay/Documents/MD/test_gemini/material_RL/labels/processed/noisy_train.csv"
IMG_DIR = "/home/akshay/Documents/MD/test_gemini/material_RL/image/all"
obj = 'rejection'
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_resnet18_512_rej.ckpt")

# --- MLP Definition ---
class MLP(nn.Module):
    def __init__(self, in_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# --- Data Loading and Preprocessing ---
def create_dataset():
    df = pd.read_csv(LABELS_CSV)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    # Remove the final fully connected layer to get features
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    labels_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_path = os.path.join(IMG_DIR, row['index'] + '.png')
        if not os.path.exists(img_path):
            continue
        
        img = Image.open(img_path).convert('RGB')
        img_t = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            features = resnet(img_t)
            features = features.view(features.size(0), -1) # Flatten
            features_list.append(features)
            labels_list.append(torch.tensor([row['flux']], dtype=torch.float32))

    return TensorDataset(torch.cat(features_list), torch.cat(labels_list))

# --- Training Loop ---
def train_model():
    dataset = create_dataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLP(in_size=512)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for features, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), NEW_MODEL_PATH)
    print(f"Model trained and saved to {NEW_MODEL_PATH}")

if __name__ == "__main__":
    train_model()
