import os
import pandas as pd
import torch 
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

class FashionDataset(Dataset):
    def __init__(
        self, csv_file, img_dir, column_class="articleTypeId", transform=None
    ):
        """
        Args:
            csv_file (str): Path to the CSV file with labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir 
        self.transform = transform 
        self.targets = list(self.df[column_class].values)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir, f"{self.df.loc[idx,'imageId']}.jpg"
        ) 
        image = Image.open(img_name).convert("RGB")  

        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx]
