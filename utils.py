import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold


def prepare_data(dataset_dir, csv_dir):
    # Read X-ray paths from the five KL grade folders.
    images = []
    for grade in range(5):
        folder = os.path.join(dataset_dir, str(grade))
        if not os.path.isdir(folder):
            raise FileNotFoundError('Missing dataset folder: ' + folder)
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                images.append((os.path.abspath(path), grade))

    df = pd.DataFrame(images, columns=['image_path', 'KL_grade'])
    if df.empty:
        raise ValueError('No X-ray images found')

    train_val, test = train_test_split(df, test_size=0.20,
                                       random_state=42, stratify=df['KL_grade'])
    train, val = train_test_split(train_val, test_size=0.20,
                                  random_state=42, stratify=train_val['KL_grade'])
    os.makedirs(csv_dir, exist_ok=True)

    for name, data in [('full', df), ('training', train),
                       ('validation', val), ('test', test)]:
        data.to_csv(os.path.join(csv_dir, name + '.csv'), index=False)
        counts = data['KL_grade'].value_counts().reindex(range(5), fill_value=0)
        plt.figure()
        plt.bar(counts.index, counts.values)
        plt.xticks(range(5))
        plt.xlabel('KL Grade')
        plt.ylabel('Number of Images')
        plt.title('KL Grade Distribution - ' + name)
        plt.tight_layout()
        plt.savefig(os.path.join(csv_dir, name + '.png'))
        plt.close()

    # Five folds from the remaining 80%; test data stays separate.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, train_val['KL_grade'])):
        for name, data in [('train', train_val.iloc[train_idx]),
                           ('val', train_val.iloc[val_idx])]:
            data.to_csv(os.path.join(csv_dir, f'{name}_fold_{fold}.csv'), index=False)
            counts = data['KL_grade'].value_counts().reindex(range(5), fill_value=0)
            plt.figure()
            plt.bar(counts.index, counts.values)
            plt.xticks(range(5))
            plt.xlabel('KL Grade')
            plt.ylabel('Number of Images')
            plt.title(f'{name.title()} - Fold {fold}')
            plt.tight_layout()
            plt.savefig(os.path.join(csv_dir, f'{name}_fold_{fold}.png'))
            plt.close()

    print('Dataset images:', len(df))
    print('Test images:', len(test))
    print('CSV files and charts saved to:', csv_dir)
