# Knee Osteoarthritis KL Grade Classification

## Project Description

This project uses deep learning to classify knee X-ray images into five Kellgren-Lawrence (KL) grades, from 0 to 4. The model is trained using a pretrained ResNet18 model in PyTorch.

## Dataset

Dataset: [OSAIL Knee Osteoarthritis KL Scoring Dataset](https://www.kaggle.com/datasets/peymannejat/osail-knee-osteoarthritis-kl-scoring-dataset)

The dataset contains 2,352 labeled knee X-ray images.

- 80% of the images are used for training and validation.
- 20% of the images are kept for testing.
- 5-fold stratified cross-validation is used.

Dataset folder: `OSAIL_KL_Dataset/Labeled`

## Project Files

- `args.py` – Training settings
- `main.py` – Runs training or testing
- `dataset.py` – Loads and transforms X-ray images
- `model.py` – Defines the ResNet model
- `utils.py` – Splits the dataset and creates CSV files and charts
- `trainer.py` – Trains the model and saves results
- `evaluate.py` – Evaluates the trained model

## Training Settings

- Model: ResNet18
- Epochs: 10 per fold
- Batch size: 16
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Cross-Entropy Loss
- Cross-validation: 5 folds

The best model from each fold is saved based on validation balanced accuracy.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

In `main.py`:

- `train = True` trains all five folds.
- `train = False` evaluates the previously saved model.

## Results

The model was trained for 10 epochs on each of the five folds.

| Metric | Result |
|---|---|
| Mean validation balanced accuracy | 52.93% |
| Test accuracy | 61.15% |
| Test balanced accuracy | 57.71% |
| Test macro F1-score | 54.40% |

The final evaluation used 471 test images.

The model performed better on KL grades 0 and 3 but had difficulty classifying KL grade 1.

## Conclusion

This project shows how ResNet18 can be used to classify knee osteoarthritis severity from X-ray images. The model achieved 61.15% test accuracy. The results show that some KL grades are harder to classify than others, so further improvements could help the model perform better.