
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import get_args
from model import MyModel
from dataset import MyDataset
from trainer import train_model
from utils import prepare_data
from evaluate import evaluate_model


def main():
    args = get_args()

    # Set to False because all 5 folds are already trained.
    train = False

    if train:
        prepare_data(args.dataset_dir, args.csv_dir)
        print('Using device:', 'CUDA' if torch.cuda.is_available() else 'CPU')

        scores = []

        for fold in range(5):
            print('\nTraining fold', fold)

            train_df = pd.read_csv(
                os.path.join(args.csv_dir, f'train_fold_{fold}.csv')
            )
            val_df = pd.read_csv(
                os.path.join(args.csv_dir, f'val_fold_{fold}.csv')
            )

            train_loader = DataLoader(
                MyDataset(train_df, training=True),
                batch_size=args.bs,
                shuffle=True
            )
            val_loader = DataLoader(
                MyDataset(val_df),
                batch_size=args.bs
            )

            model = MyModel(args.backbone)
            score = train_model(model, train_loader, val_loader, fold, args)
            scores.append(score['balanced_accuracy'])

        print('Validation balanced accuracy for each fold:', scores)
        print('Mean balanced accuracy:', sum(scores) / len(scores))

        best_fold = scores.index(max(scores))

    else:
        # Fold 4 had the best validation balanced accuracy.
        best_fold = 4

    checkpoint = os.path.join(
        args.output_dir, 'out', f'best_model_fold_{best_fold}.pth'
    )
    test_csv = os.path.join(args.csv_dir, 'test.csv')

    print('\nEvaluating fold', best_fold)
    evaluate_model(checkpoint, test_csv, args.backbone, args.bs)


if __name__ == '__main__':
    main()
