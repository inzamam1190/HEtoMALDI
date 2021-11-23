import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_lightning as pl

# from torchvision import transforms
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

    logger = SummaryWriter(flush_secs=10)

    best_acc = 0.0
    total_steps = 0

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0
            nproc = 0

            means = torch.zeros(3, device=device)
            vs = torch.zeros(3, device=device)

            # iterate over data
            itbar = tqdm(dataloader, position=0, leave=True, desc=f"{phase} ep {epoch}")
            for x, y in itbar:
                x = x.to(device)
                y = y.to(device)
                step += 1

                y = torch.round(y)

                if False:
                    # running computation of mean and variance of each channel
                    means = (means * (step - 1) + x.mean(dim=(0, -2, -1))) / step
                    vs = (vs * (step - 1) + x.var(dim=(0, -2, -1))) / step
                    print("X mean", means)
                    print("X std", torch.sqrt(vs))
                    print("Y mean", y.mean(dim=(-2, -1)))

                # forward pass
                if phase == "train":
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)["out"]
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)["out"]
                        loss = loss_fn(outputs, y)

                nbatch = x.shape[0]
                nproc += nbatch
                # stats - whatever is the phase
                with torch.no_grad():
                    acc = acc_fn(outputs, y)

                    running_acc += acc.item() * nbatch
                    running_loss += loss.item() * nbatch

                itbar.set_postfix(
                    loss=loss.item(),
                    acc=acc.item(),
                    running_acc=running_acc / nproc,
                    running_loss=running_loss / nproc,
                )

                if phase == "train":
                    logger.add_scalar(f"loss/{phase}", loss.item(), total_steps)
                    logger.add_scalar(f"acc/{phase}", acc.item(), total_steps)
                    total_steps += 1
            epoch_loss = running_loss / nproc
            epoch_acc = running_acc / nproc

            train_loss.append(epoch_loss) if phase == "train" else valid_loss.append(
                epoch_loss
            )
            train_acc.append(epoch_acc) if phase == "train" else valid_acc.append(
                epoch_acc
            )
            if phase != "train":
                logger.add_scalar(f"loss/{phase}", epoch_loss, epoch)
                logger.add_scalar(f"acc/{phase}", epoch_acc, epoch)

            print("{} Loss: {:.4f} Acc: {}".format(phase, epoch_loss, epoch_acc))

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
        plt.savefig("output/" + f"plot_epoch{epoch}.png")

        # save the model for each epoch
        torch.save(model.state_dict(), "output/" + f"model_epoch{epoch}.pt")

    time_elapsed = time.time() - start
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return None


def acc_metric(predb, yb):
    # p = torch.sigmoid(predb)
    predmask = (predb > 0.0).double()
    return (predmask == yb).float().mean()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):

    model_ft = models.segmentation.fcn_resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[4] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    model_ft.aux_classifier[4] = nn.Conv2d(
        256, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    input_size = 512

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name, param.data.numel())
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    return model_ft, input_size, params_to_update


if __name__ == "__main__":
    pl.seed_everything(0)
    model_ft, input_size, params_to_update = initialize_model(
        1, feature_extract=True, use_pretrained=True
    )
    # summary(unet)
    # use torchinfo to see model summary
    dm = czi_dataloader.CZIDataModule("/achs/inzamam/msi/czi_data/dataset")
    dm.setup()
    train_dl = dm.train_dataloader(batch_size=16)
    valid_dl = dm.val_dataloader()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.0])).to("cuda")
    opt = torch.optim.Adam(params_to_update, lr=1e-3, amsgrad=True)
    _ = train(model_ft, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50)
