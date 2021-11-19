import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import h5py
from tqdm import tqdm
from patchify import patchify, unpatchify
import time
from torchvision import models
import matplotlib.pyplot as plt


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
    
    return model_ft

def get_prediction(patches,model,device):
    pred_mask = np.empty(shape=(patches.shape[0],patches.shape[1],512,512), dtype=np.float32)
    for i in tqdm(range(patches.shape[0])):
        for j in range(patches.shape[1]):
            with torch.no_grad():
                image = torch.from_numpy(patches[i,j,0,:,:,:])
                image = torch.permute(image, (2, 0, 1))
                img = transform(image) # unsqueeze and pass to model 
                img = img.to(device)
                img = img.unsqueeze(axis=0)
                predb = model(img)['out']
                predmask = torch.sigmoid(predb).squeeze().cpu().detach().numpy()
                np.append(pred_mask[i,j,:,:],predmask)
    return pred_mask

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--statedict_file', required=True, help='Filename to load')
    parser.add_argument('--section_id', required=True, help='Subject ID, i.e. 1900')
    parser.add_argument('--patch_size', default=2048, type=int, help='Size of square patches in pixels (not counting halos).')
    parser.add_argument('--halo_width', default=0, type=int, help='Size of halos to add to patch size on each side.')
    parser.add_argument('--output_file', '-o', required=True, help='Where to place output torch tensor (should end in .pt or .pth)')
    args = parser.parse_args()

    model = initialize_model(1, feature_extract=True)

    # '/achs/inzamam/msi/model_epoch12.pt'
    model.load_state_dict(torch.load(args.statedict_file))
    model.eval()

    device = torch.device("cuda:0")
    model = model.to(device)

    tx = transforms.Normalize(mean=[696.5508, 479.1579, 525.0390], std=[247.2921, 204.0386, 174.9929])

    M = args.patch_size

    ffile = '/achs/inzamam/msi/czi_features.h5'
    with h5py.File(ffile,'r') as f:
        img = f[f'features_{args.section_id}']

        with torch.no_grad():
            fullpred = torch.zeros(*img.shape[-3:-1], dtype=torch.float32, device='cpu')
            for i in tqdm(range(0,
                    fullpred.shape[0] - args.patch_size + 1,
                    args.patch_size)):
                for j in range(0,
                        fullpred.shape[1] - args.patch_size + 1,
                        args.patch_size):
                    # Load patch with halo
                    starty = max(i - args.halo_width, 0)
                    startx = max(j - args.halo_width, 0)
                    endy = min(i + M + args.halo_width, img.shape[-3])
                    endx = min(j + M + args.halo_width, img.shape[-2])
                    # compute interior bounds
                    intstarty = i
                    intstartx = j
                    intendy = min(i + M, img.shape[-3])
                    intendx = min(j + M, img.shape[-2])

                    patch = torch.tensor(img[starty:endy, startx:endx, :]).cuda()
                    patch = torch.permute(patch, (2, 0, 1)).unsqueeze(0)
                    # standardize
                    patch = tx(patch)
                    # compute prediction
                    pred = model(patch)['out']
                    predprob = torch.sigmoid(pred).squeeze().cpu()
                    fullpred[
                            intstarty:intendy,
                            intstartx:intendx,
                        ] = \
                            predprob[
                                intstarty - starty: predprob.shape[0] + intendy - endy,
                                intstartx - startx: predprob.shape[1] + intendx - endx,
                            ]


    torch.save(fullpred, args.output_file)
