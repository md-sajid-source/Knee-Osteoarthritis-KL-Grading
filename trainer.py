
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from args import *
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize




# Added: calculate the required metrics
def calculate_metrics(labels, outputs):
    """Return balanced accuracy, multiclass one-vs-rest ROC-AUC, macro AP."""
    from sklearn.preprocessing import label_binarize
    labels = np.asarray(labels, dtype=int)
    probabilities = torch.softmax(outputs.detach().cpu(), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    if len(np.unique(labels)) != probabilities.shape[1]:
        return balanced_acc, float('nan'), float('nan')
    roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro', labels=np.arange(probabilities.shape[1]))
    targets = label_binarize(labels, classes=np.arange(probabilities.shape[1]))
    avg_precision = average_precision_score(targets, probabilities, average='macro')
    return balanced_acc, roc_auc, avg_precision


def train_model(model, train_loader, val_loader, fold=0, args=None, output_dir=None):
    args = args or get_args()

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    # Added: store metrics for plotting
    train_losses, val_losses = [], []
    train_bal_accs, val_bal_accs = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_avg_precs, val_avg_precs = [], []

    output_dir = output_dir or os.path.join(args.output_dir, "out")
    os.makedirs(output_dir, exist_ok=True)

    best_balanced_acc = -1

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0
        train_labels, train_outputs = [], []

        for batch in train_loader:
            inputs = batch['xray'].to(device)
            labels = batch['target'].to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            train_labels.extend(labels.cpu().numpy())
            train_outputs.append(outputs.detach().cpu())

        train_loss = running_loss / len(train_loader.dataset)
        train_outputs = torch.cat(train_outputs)

        train_metrics = calculate_metrics(
            np.array(train_labels), train_outputs
        )

        # Added: calculate validation metrics
        val_loss, val_metrics = validate_model(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_bal_accs.append(train_metrics[0])
        val_bal_accs.append(val_metrics[0])


        train_roc_aucs.append(train_metrics[1])
        val_roc_aucs.append(val_metrics[1])

        train_avg_precs.append(train_metrics[2])
        val_avg_precs.append(val_metrics[2])

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("Training Loss:", round(train_loss, 4))
        print("Validation Loss:", round(val_loss, 4))
        print("Validation Balanced Accuracy:", round(val_metrics[0], 4))
        print("Validation ROC-AUC:", round(val_metrics[1], 4))
        print("Validation Average Precision:", round(val_metrics[2], 4))

        # Added: save model with the best validation Balanced Accuracy
        if np.isfinite(val_metrics[0]) and val_metrics[0] > best_balanced_acc:
            best_balanced_acc = val_metrics[0]

            torch.save(
                model.state_dict(),
                os.path.join(
                    output_dir, f"best_model_fold_{fold}.pth"
                )
            )

    # Added: plot training and validation metrics
    metrics = [
        ("Loss", train_losses, val_losses),
        ("Balanced Accuracy", train_bal_accs, val_bal_accs),
        ("ROC-AUC", train_roc_aucs, val_roc_aucs),
        ("Average Precision", train_avg_precs, val_avg_precs)
    ]

    for name, train_values, val_values in metrics:
        plt.figure()
        plt.plot(range(1, len(train_values) + 1), train_values, label="Training")
        plt.plot(range(1, len(val_values) + 1), val_values, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"Fold {fold} - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"fold_{fold}_{name.lower().replace(' ', '_')}.png"
            )
        )
        plt.close()

    print(f"Best validation Balanced Accuracy: {best_balanced_acc:.4f}")

    return {"balanced_accuracy": float(best_balanced_acc)}


def validate_model(model, val_loader, criterion, device):

    model.eval()
    running_loss = 0.0
    labels_all, outputs_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['xray'].to(device)
            labels = batch['target'].to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * len(labels)

            labels_all.extend(labels.cpu().numpy())
            outputs_all.append(outputs.cpu())

    loss = running_loss / len(val_loader.dataset)
    outputs_all = torch.cat(outputs_all)

    metrics = calculate_metrics(np.array(labels_all), outputs_all)

    return loss, metrics

