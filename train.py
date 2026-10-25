import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import joblib

# Import your architecture strictly from model.py
from model import ModelArchitecture

class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        labels_path = os.path.join(root_dir, 'labels.json')
        if not os.path.exists(labels_path):
            print(f"ERROR: Cannot find {labels_path}")
            return

        with open(labels_path, 'r') as f:
            self.class_to_idx = json.load(f)
            
        # The JSON keys are the index numbers (strings) and values are the bird names
        for idx_str, class_name in self.class_to_idx.items():
            label_idx = int(idx_str)
            class_dir = None
            
            # Match the bird name from the JSON to the folder name (ignoring numbers like "00_")
            for folder_name in os.listdir(root_dir):
                # Strictly ignore the new validation folder
                if folder_name == "validation":
                    continue
                    
                if class_name in folder_name and os.path.isdir(os.path.join(root_dir, folder_name)):
                    class_dir = os.path.join(root_dir, folder_name)
                    break
            
            if class_dir is None:
                print(f"Missing folder for bird: {class_name}")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def main():
    # Setup dynamic relative path to comply with absolute path restrictions
    current_script_folder = os.path.dirname(__file__)
    dataset_path = os.path.join(current_script_folder, "..", "..", "dataset")
    
    baseline_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Randomly flip the image horizontally
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly alter lighting and color context to prevent background memorization
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # Randomly shift, rotate, and scale the bird
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    full_dataset = BirdDataset(root_dir=dataset_path, transform=baseline_transform)
    
    if len(full_dataset) == 0:
        print("No images found. Please check your dataset directory structure.")
        return

    # Train on the ENTIRE dataset for maximum robustness
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelArchitecture(num_classes=20).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print an update every 50 batches so you know it is running
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Current Loss: {loss.item():.4f}")
            
        print(f"--- End of Epoch {epoch+1} | Average Loss: {running_loss/len(train_loader):.4f} ---")

    # Move model to CPU and save exactly to weights.joblib
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, "weights.joblib")
    print("Training complete. Weights saved to weights.joblib.")

if __name__ == "__main__":
    main()