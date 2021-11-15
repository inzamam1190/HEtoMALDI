import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import h5py
from tqdm import tqdm
import hdf5plugin
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
    f = h5py.File('czi_features.h5','r')
    image_1900_masked = f['features_1900']

    print(image_1900_masked )

    print('it will take some time to load the whole image in a numpy array before making patches')
    img = np.array(image_1900_masked)

    print('Done loading into array! Making patches!')
    patches_image = patchify(img, (512, 512, 3), step=512)

    print(f'Done patching! Shape of the patch: {patches_image.shape}')

    model = initialize_model(1,feature_extract=True)

    model.load_state_dict(torch.load('/achs/inzamam/msi/model_epoch12.pt'))
    model.eval()

    device = torch.device("cuda:0")
    model = model.to(device)

    pred_mask = get_prediction(patches_image,model,device)

    print(pred_mask.shape)

            