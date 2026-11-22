import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from dataset import MyDataset
from model import MyModel


def evaluate_model(checkpoint, test_csv, backbone='resnet18', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    test_loader = DataLoader(MyDataset(pd.read_csv(test_csv)), batch_size=batch_size)

    actual, predicted = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['xray'].to(device))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            actual.extend(batch['target'].tolist())

    results = {
        'accuracy': accuracy_score(actual, predicted),
        'balanced_accuracy': balanced_accuracy_score(actual, predicted),
        'macro_f1': f1_score(actual, predicted, average='macro', zero_division=0)
    }
    print('Test results:', results)
    print('Confusion matrix:\n', confusion_matrix(actual, predicted, labels=range(5)))
    print(classification_report(actual, predicted, labels=range(5), zero_division=0))
    return results
