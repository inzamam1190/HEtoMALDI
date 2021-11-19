import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
from tqdm import tqdm
from torch import nn
from torchinfo import summary
from torchvision import models

# Need to read more on hooks
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', '-d', nargs=2, required='True',
            help='Hdf5 File to extract features from followed by the dataset name')
    parser.add_argument('--target_file', '-t', nargs=2, required='True',
            help="HDF5 filename followed by dataset name to write. Must not exist.")
    parser.add_argument('--patch_size', '-p', required='True',
            help="patchsize from input image for passing to the model to extract features")

    args = parser.parse_args()

    # define model
    model = models.segmentation.fcn_resnet101(pretrained=True)
    #print model
    print(summary(model))

    file, dataset = args.datapath
    target_file, target_dataset = args.target_file
    M = int(args.patch_size)
    
    # register hook to a layer. can always change the layer. This line will change depending one the model
    model.backbone.layer2[3].conv3.register_forward_hook(get_features('feats')) #shape(c,h,w)-->(c,64,64)
    device = torch.device("cuda:0")
    model = model.to(device)

    # test if hook works
    features={}
    random_input = torch.zeros((1,3,512,512), dtype=torch.float32, device='cuda:0')
    out = model(random_input)['out']
    fts = features['feats'].squeeze().cpu().numpy()
    print(fts.shape) # should be (M,64,64)
    del features

    with h5py.File(file, "r") as fsource:
        ds_source = fsource[dataset]
        with h5py.File(target_file, "a") as ftarget:
            if target_dataset in ftarget:
                raise RuntimeError(
                    f"Refusing to overwriting existing dataset {target_dataset} in {target_file}"
                )
            ds_target = ftarget.create_dataset(
                target_dataset, shape=(M,ds_source.shape[0]//(M/fts.shape[1]),ds_source.shape[1]//(M/fts.shape[2])), chunks=True, dtype='f'
                )

            for i in tqdm(range(0, ds_source.shape[0], M)):
                for j in range(0, ds_source.shape[1], M):
                    # Load patch
                    features = {} 
                    endy = min(i+M, ds_source.shape[0])
                    endx = min(j+M, ds_source.shape[1])
                    patch = torch.tensor(ds_source[i:endy, j:endx, :]).cuda()
                    patch = torch.permute(patch, (2, 0, 1)).unsqueeze(0)
                    out = model(patch)['out']
                    ft = features['feats'].squeeze().cpu().numpy()
                    ds_target[:, i:endy//(M//ft.shape[1]), j:endx//(M//ft.shape[2])] = ft
                    del features



    


