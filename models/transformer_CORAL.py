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


# NEW CORAL CODE:
# Convert class labels 0, 1, 2, 3 into ordinal CORAL targets.
# Example:
# 0 -> [0, 0, 0]
# 1 -> [1, 0, 0]
# 2 -> [1, 1, 0]
# 3 -> [1, 1, 1]
def labels_to_coral(labels, num_classes):
    levels = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > levels).float()


# NEW CORAL CODE:
# Convert CORAL logits into final class predictions.
# The model outputs 3 logits: [>=1, >=2, >=3].
# We apply sigmoid, threshold at 0.5, then sum the passed thresholds.
def coral_logits_to_labels(outputs):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).sum(dim=1)
    return preds


def main():
    data_dir = "../dataset"
    batch_size = 16
    num_classes = 4
    num_epochs = 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Swin Vision Transformer
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

    # DELETED/REPLACED BASELINE CODE:
    # model.head = nn.Linear(model.head.in_features, num_classes)

    # NEW CORAL CODE:
    # CORAL uses num_classes - 1 outputs.
    # For MOAKS labels 0-3, this gives 3 threshold logits:
    # [severity >= 1, severity >= 2, severity >= 3]
    model.head = nn.Linear(model.head.in_features, num_classes - 1)

    model = model.to(device)

    # DELETED/REPLACED BASELINE CODE:
    # criterion = nn.CrossEntropyLoss()

    # NEW CORAL CODE:
    # CORAL trains each threshold as a binary classification problem.
    # BCEWithLogitsLoss expects raw logits and internally applies sigmoid.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # DELETED/REPLACED BASELINE CODE:
            # loss = criterion(outputs, labels)

            # NEW CORAL CODE:
            # Convert integer labels into ordinal threshold vectors.
            # Example: label 2 becomes [1, 1, 0].
            coral_targets = labels_to_coral(labels, num_classes)
            loss = criterion(outputs, coral_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # DELETED/REPLACED BASELINE CODE:
            # preds = torch.argmax(outputs, dim=1)

            # NEW CORAL CODE:
            # Convert threshold logits into class predictions.
            preds = coral_logits_to_labels(outputs)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                # DELETED/REPLACED BASELINE CODE:
                # preds = torch.argmax(outputs, dim=1)

                # NEW CORAL CODE:
                preds = coral_logits_to_labels(outputs)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # NEW CORAL CODE:
            # Save CORAL version separately so it does not overwrite baseline.
            torch.save(model.state_dict(), "best_swin_t_coral_bml.pth")

    print("Best validation accuracy:", best_val_acc)

    # DELETED/REPLACED BASELINE CODE:
    # model.load_state_dict(torch.load("best_swin_t_bml.pth", map_location=device))

    # NEW CORAL CODE:
    model.load_state_dict(torch.load("best_swin_t_coral_bml.pth", map_location=device))

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            # DELETED/REPLACED BASELINE CODE:
            # preds = torch.argmax(outputs, dim=1)

            # NEW CORAL CODE:
            preds = coral_logits_to_labels(outputs)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    hyperparams = {
    "batch_size": batch_size,
    "num_classes": num_classes,
    "num_outputs": num_classes - 1,
    "num_epochs": num_epochs,
    "learning_rate": lr,
    "optimizer": "Adam",
    "loss": "BCEWithLogitsLoss",
    "coral_thresholds": ["severity >= 1", "severity >= 2", "severity >= 3"],
    "prediction_rule": "sigmoid(outputs) > 0.5, then sum thresholds",
    "pretrained_weights": "Swin_T_Weights.DEFAULT",
    "image_size": "224x224",
    "augmentation": ["RandomHorizontalFlip", "RandomRotation(10)"],
    "normalization": "ImageNet mean/std",
    }

    save_results_json(
    output_path="results_swin_t_coral.json",
    model_name="Swin-T",
    loss_name="CORAL",
    best_val_acc=best_val_acc,
    all_labels=all_labels,
    all_preds=all_preds,
    hyperparams=hyperparams
    )


if __name__ == "__main__":
    main()