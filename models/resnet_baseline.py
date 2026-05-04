import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error
)

# HELPER OUTPUT FUNCTION TO SAVE RESULTS IN JSON FORMAT
def save_results_json(
    output_path,
    model_name,
    loss_name,
    best_val_acc,
    all_labels,
    all_preds,
    hyperparams
):
    test_acc = accuracy_score(all_labels, all_preds)
    test_mae = mean_absolute_error(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    ordinal_errors = [abs(int(t) - int(p)) for t, p in zip(all_labels, all_preds)]

    ordinal_error_breakdown = {
        "exact_match": sum(e == 0 for e in ordinal_errors),
        "off_by_1": sum(e == 1 for e in ordinal_errors),
        "off_by_2": sum(e == 2 for e in ordinal_errors),
        "off_by_3": sum(e == 3 for e in ordinal_errors),
    }

    total = len(ordinal_errors)
    ordinal_error_percentages = {
        key: value / total if total > 0 else 0.0
        for key, value in ordinal_error_breakdown.items()
    }

    results = {
        "model_name": model_name,
        "loss_name": loss_name,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_mae": float(test_mae),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "ordinal_error_breakdown_counts": ordinal_error_breakdown,
        "ordinal_error_breakdown_percentages": ordinal_error_percentages,
        "true_labels": [int(x) for x in all_labels],
        "pred_labels": [int(x) for x in all_preds],
        "hyperparameters": hyperparams,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to {output_path}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print("Ordinal Error Breakdown:", ordinal_error_breakdown)

def main():
    data_dir = "../dataset"
    batch_size = 16
    num_classes = 4
    num_epochs = 20
    lr = 1e-4
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

    # Get the resnet model, plus loss and optimizer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update loss/predictions
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
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
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | ")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            # Save if it has best so far accuracy
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet18_bml.pth")
    print("Best validation accuracy:", best_val_acc)

    # Testing best model
    model.load_state_dict(torch.load("best_resnet18_bml.pth"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    hyperparams = {
    "batch_size": batch_size,
    "num_classes": num_classes,
    "num_epochs": num_epochs,
    "learning_rate": lr,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss",
    "pretrained_weights": "ResNet18_Weights.DEFAULT",
    "image_size": "224x224",
    "augmentation": ["RandomHorizontalFlip", "RandomRotation(10)"],
    "normalization": "ImageNet mean/std",
    }
    save_results_json(
    output_path="results_resnet18_ce.json",
    model_name="ResNet18",
    loss_name="CrossEntropy",
    best_val_acc=best_val_acc,
    all_labels=all_labels,
    all_preds=all_preds,
    hyperparams=hyperparams
    )

if __name__ == '__main__':
    main()
