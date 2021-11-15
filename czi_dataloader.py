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


class CZIPatchDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir 
        self.transform = transform
        self.filenames = sorted([name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, name))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        #img_path = os.listdir(self.img_dir)[idx]
        dict_patch = torch.load(os.path.join(self.img_dir,self.filenames[idx]))
        # get patch
        image = dict_patch.get('data') 
        # change dimension from HWC to CHW 
        image = torch.permute(image, (2, 0, 1))
        # get label
        label = dict_patch.get('label').unsqueeze(dim=0)
        if self.transform:
            image = self.transform(image)
        return image, label

class CZIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    def setup(self):

        # Use train_test_split from filenames and split the data
        data_full = CZIPatchDataset(img_dir=self.data_dir, transform=self.transform) #get the full dataset
        self.train, self.val, self.test = random_split(data_full, [5000, 1000, 1225], generator=torch.Generator().manual_seed(42))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=16, shuffle=True, pin_memory=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=16, shuffle=True, pin_memory=True, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=16, shuffle=True, pin_memory=True, num_workers=10)
                                             

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', '-d', required='True',
            help='data directory for the CZI patches')
    args = parser.parse_args()
    
    #dataset = CZIPatchDataset(dir)
    dm = CZIDataModule(args.datapath)
    dm.setup()
    
    train_features, train_labels = next(iter(dm.train_dataloader()))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    print(train_features[0])
    print(train_labels[0])
    

