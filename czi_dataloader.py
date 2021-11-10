import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.model_selection import KFold, train_test_split


# Dataset class that can take input all the slices e.g 1327,1405,1900 etc.
class CZIPatchDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir 
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, name))])

    def __getitem__(self, idx):
        img_path = os.listdir(self.img_dir)[idx]
        dict_patch = torch.load(os.path.join(self.img_dir,img_path))
        # get patch
        image = dict_patch.get('data') 
        # normalize to range 0-1
        image -= image.min(1, keepdim=True)[0]
        image /= image.max(1, keepdim=True)[0]
        # change dimension from HWC to CHW 
        image = torch.permute(image, (2, 0, 1))
        # get label
        label = dict_patch.get('label')
        if self.transform:
            image = self.transform(image)
        return image, label

class CZIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        #self.dims = (3, 512, 512)

    def setup(self):

        data_full = CZIPatchDataset(self.data_dir) #get the full dataset
        self.train, self.val, self.test = random_split(data_full, [6605, 1000, 2000], generator=torch.Generator().manual_seed(42))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', '-d', required='True',
            help='data directory for the CZI patches')
    args = parser.parse_args()
    
    #dataset = CZIPatchDataset(dir)
    dm = CZIDataModule(args.datapath)
    dm.setup()
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(dm.train_dataloader()))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    print(train_features[0])
    print(train_labels[0])
    

