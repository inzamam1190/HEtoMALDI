import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
#from torchvision import transforms
from torch.utils.data import random_split
from sklearn.model_selection import KFold, train_test_split
import time
import czi_dataloader
from torch import nn
from torchinfo import summary

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    # define as forward not call
    def forward(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1
    
    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
                            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    device = torch.device("cuda:0")
    start = time.time()
    model.to(device)
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
    
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in tqdm(dataloader,position=0, leave=True):
                x = x.to(device)
                y = y.to(device)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(torch.squeeze(outputs), y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(torch.squeeze(outputs), y)

                # stats - whatever is the phase
                with torch.no_grad():
                    acc = acc_fn(outputs, y)

                    running_acc  += acc*dataloader.batch_size
                    running_loss += loss*dataloader.batch_size 
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            train_acc.append(epoch_acc) if phase=='train' else valid_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

        # plot the training loss
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(train_loss, label="train_loss")
        plt.plot(valid_loss, label="validation_loss")
        plt.plot(train_acc, label="train_accuracy")
        plt.plot(valid_acc, label="validation_accuracy")
        plt.title("Training Loss/Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.savefig('/home/64f/msi/output/plot/' + f'plot_epoch{epoch}.png')

        #save the model for each epoch
        torch.save(model.state_dict(), '/home/64f/msi/output/saved_model' + f'model_epoch{epoch}.pt')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

    return None    

def acc_metric(predb, yb):
    p = torch.sigmoid(predb)
    predmask = (p > 0.5).double()
    return (predmask == yb).float().mean()

if __name__ == '__main__':
    unet = UNET(3,1)
    summary(unet)
    # use torchinfo to see model summary 
    dm = czi_dataloader.CZIDataModule('/achs/inzamam/msi/czi_data/dataset')
    dm.setup()
    train_dl = dm.train_dataloader()
    valid_dl = dm.val_dataloader()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.01)
    _ = train(unet,train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50)
    


