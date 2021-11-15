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
from torchvision import models


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
                model.train()  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.eval()  # Set model to evaluate mode
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
                    outputs = model(x)['out']
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)['out']
                        loss = loss_fn(outputs, y)

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
        plt.figure()
        plt.plot(train_loss, label="train_loss")
        plt.plot(valid_loss, label="validation_loss")
        plt.plot(train_acc, label="train_accuracy")
        plt.plot(valid_acc, label="validation_accuracy")
        plt.title("Training Loss/Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.savefig('/home/64f/msi/output/plot1/' + f'plot_epoch{epoch}.png')

        #save the model for each epoch
        torch.save(model.state_dict(), '/home/64f/msi/output/saved_model1/' + f'model_epoch{epoch}.pt')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

    return None    

def acc_metric(predb, yb):
    #p = torch.sigmoid(predb)
    predmask = (predb > 0.0).double()
    return (predmask == yb).float().mean()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):

    model_ft = models.segmentation.fcn_resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    input_size = 512

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    return model_ft, input_size, params_to_update

if __name__ == '__main__':
    model_ft, input_size, params_to_update = initialize_model(1, feature_extract=True, use_pretrained=True)
    #summary(unet)
    # use torchinfo to see model summary 
    dm = czi_dataloader.CZIDataModule('/achs/inzamam/msi/czi_data/dataset')
    dm.setup()
    train_dl = dm.train_dataloader()
    valid_dl = dm.val_dataloader()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(params_to_update, lr=0.01)
    _ = train(model_ft,train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50)
    


