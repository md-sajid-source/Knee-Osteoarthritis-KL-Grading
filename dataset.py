import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data, training=False):
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        steps = [transforms.Resize((224, 224))]
        if training:
            steps.append(transforms.RandomRotation(8))
            steps.append(transforms.RandomHorizontalFlip())
        steps += [transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(row['image_path']) as image:
            xray = self.transform(image.convert('RGB'))
        return {'xray': xray, 'target': int(row['KL_grade'])}
