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
from torchvision import transforms

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
    parser.add_argument(
        "--patch_size", 
        default=2048,
        type=int,
        help="Size of square patches in pixels (not counting halos).")
    parser.add_argument(
        "--halo_width", 
        default=0,
        type=int,
        help="Size of halos to add to patch size on each side.")


    args = parser.parse_args()

    # define model
    model = models.segmentation.fcn_resnet101(pretrained=True)
    #print model
    summary(model)

    file, dataset = args.datapath
    target_file, target_dataset = args.target_file
    M = args.patch_size
    H = args.halo_width
    tx = transforms.Normalize(mean=[696.5508, 479.1579, 525.0390], std=[247.2921, 204.0386, 174.9929])
    
    # register hook to a layer. can always change the layer. This line will change depending one the model
    model.backbone.layer4.register_forward_hook(get_features('feats')) #shape(c,h,w)-->(c,64,64)
    device = torch.device("cuda:0")
    model = model.to(device)

    # test if hook works
    features={}
    random_input = torch.zeros((1,3,M,M), dtype=torch.float32, device='cuda:0')
    out = model(random_input)['out']
    fts = features['feats'].squeeze().cpu().numpy()
    print(fts.shape) 
    #downsample value
    
    del features

    with h5py.File(file, "r") as fsource:
        ds_source = fsource[dataset]
        with h5py.File(target_file, "a") as ftarget:
            if target_dataset in ftarget:
                raise RuntimeError(
                    f"Refusing to overwriting existing dataset {target_dataset} in {target_file}"
                )
            div = M//fts.shape[1]
            ds_target = ftarget.create_dataset(
                target_dataset, shape=(fts.shape[0],ds_source.shape[1]//div,ds_source.shape[2]//div), chunks=True, dtype='f'
                )

            for i in tqdm(range(0, ds_source.shape[1], M)):
                for j in range(0, ds_source.shape[2], M):
                    features={}
                    
                    endy = min(i+M, ds_source.shape[1])
                    endx = min(j+M, ds_source.shape[2])
                    patch = torch.tensor(ds_source[:, i:endy, j:endx]).cuda()
                    patch = patch.unsqueeze(0)
                    #standardize
                    patch = tx(patch)

                    #get features
                    out = model(patch)['out']
                    ft = features['feats'].squeeze().cpu().numpy()

                    ds_target[:, i:endy//div, j:endx//div] = ft
                    del features
                        


    


