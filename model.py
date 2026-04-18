import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = r"C:\pythonproj\dsc140b\data"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

CITIES = ['Anaheim', 'Bakersfield', 'Los_Angeles', 'Riverside', 'San_Diego', 'SLO']

class SoCalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load all image paths and labels
all_images = []
all_labels = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith('.jpg'):
        city = fname.split('-')[0]
        if city in CITIES:
            all_images.append(os.path.join(DATA_DIR, fname))
            all_labels.append(CITIES.index(city))

print(f"Total images: {len(all_images)}")
print(f"Labels found: {set(all_labels)}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Split into train and validation sets
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Create datasets and dataloaders
train_dataset = SoCalDataset(train_imgs, train_labels, transform=train_transform)
val_dataset = SoCalDataset(val_imgs, val_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Load pretrained ResNet18 and modify final layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer with our 6-class classifier
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)

# Only optimize the final layer parameters
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("Model loaded successfully!")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if __name__ == '__main__':
    start_time = time.time()

    # Training loop
    train_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    print("Training complete!")

    # Fine-tuning - unfreeze all layers
    print("\nStarting fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    fine_tune_losses = []
    fine_tune_accuracies = []
    best_val_acc = 0.0

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        fine_tune_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        fine_tune_accuracies.append(val_acc)

        # Save only if best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), r"C:\pythonproj\dsc140b\weights.pt")
            print(f"Fine-tune Epoch [{epoch+1}/10] Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}% ← Best saved!")
        else:
            print(f"Fine-tune Epoch [{epoch+1}/10] Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Training time
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f"Total training time: {elapsed:.2f} minutes")

    # Plot training curve
    all_losses = train_losses + fine_tune_losses
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Empirical Risk (Loss)')
    plt.title('Training Curve')
    plt.axvline(x=NUM_EPOCHS, color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend()
    plt.savefig(r'C:\pythonproj\dsc140b\training_curve.png')
    plt.show()
    print("Training curve saved!")