import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torchinfo import summary
from torchvision import models

# Need to read more on hooks
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def extract(model, image):
    # define h5 file
    
    f = h5.File('deep_features.h5','w')
    f.create_dataset('features', shape=()) # What will be shape of the h5? It will depend on which layer we're extracting features from 

    # placeholder for batch features
    features = {}

    # loop through batches 
    # need to work on this

    for i in tqdm(range(0, image.shape[0], 512)):
        for j in range(0, image.shape[1], 512):
            # Load patch
            endy = min(i+512, image.shape[-3])
            endx = min(j+512, image.shape[-2])
            patch = torch.tensor(image[i:endy, j:endx, :]).cuda()
            patch = torch.permute(patch, (2, 0, 1)).unsqueeze(0) #(1,3,512,512)
           
            # forward pass [with feature extraction]
            preds = model(patch)
            
            fts = (features['feats'].cpu().numpy()) #(1,3,64,64)

            #Save fts to the h5 file

    

if __name__ == '__main__':
    # Need to add argparse args

    model = models.segmentation.fcn_resnet101(pretrained=True)
    #print model
    print(summary(model))
    #summary(unet)

    # I am just using one slice now. will make the other slice features ready by tonight and then we can import all them in a list 
    f = h5py.File('/achs/inzamam/msi/czi_features.h5','r')
    image_1900_masked = f['features_1900'] #shape(h,w,c)
    
    # register hook to a layer. can always change the layer
    model.backbone.layer2[3].conv3.register_forward_hook(get_features('feats')) #which layer it should be?

    device = torch.device("cuda:0")
    model = model.to(device)

    #extract() #


