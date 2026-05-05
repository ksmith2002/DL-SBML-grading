import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

class CoralLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        return self.fc(x) + self.bias

def labels_to_levels(labels, num_classes):
    levels = []
    for label in labels:
        levels.append([1 if label > i else 0 for i in range(num_classes - 1)])
    return torch.tensor(levels, dtype=torch.float32, device=labels.device)


def coral_loss(logits, labels, num_classes, class_weights=None):
    levels = labels_to_levels(labels, num_classes)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, levels, reduction='none')

    if class_weights is not None:
        weights = class_weights[labels].unsqueeze(1)
        loss = loss * weights

    return loss.mean()


def coral_predict(logits):
    probs = torch.sigmoid(logits)
    return torch.sum(probs > 0.5, dim=1)

# RESNET
def main():
    data_dir = "dataset_2"
    batch_size = 16
    num_classes = 4
    num_epochs = 20
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add randomization here, like flipping/rotation
    # Transforms to preprocess data (change to grayscale, resize, make to tensor, normalize)
    # The normalization is for image net normalization, what it was originally trained on
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Class mapping:", train_dataset.class_to_idx)
    print("Train counts:", Counter(train_dataset.targets))
    print("Val counts:", Counter(val_dataset.targets))
    print("Test counts:", Counter(test_dataset.targets))

    # Get the resnet model, plus loss and optimizer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = CoralLayer(in_features, num_classes)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    counts = torch.tensor([2896, 4432, 2508, 1168], dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    weights = weights.to(device)

    # Training
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            # Load in images
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Results
            outputs = model(images)
            loss = coral_loss(outputs, labels, num_classes, class_weights=weights)
            loss.backward()
            optimizer.step()
            # Update loss/predictions
            running_loss += loss.item() * images.size(0)
            preds = coral_predict(outputs)
            # Check for accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        # Similar to above but no_grad
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = coral_predict(outputs)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | ")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            # Save if it has best so far accuracy
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet18_bml_coral_weighted_2.pth")
    print("Best validation accuracy:", best_val_acc)

    # Testing best model
    model.load_state_dict(torch.load("best_resnet18_bml_coral_weighted_2.pth"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = coral_predict(outputs)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("Prediction counts:", Counter(all_preds))
    print("True label counts:", Counter(all_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == '__main__':
    main()