import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from tqdm import tqdm

def save_patches(data_array, mask_array, path:str, imagenumber:str):
    for i in tqdm(range(data_array.shape[0])):  
        if data_array[i].mean() == 0.0:
            continue
        # convert both patches and labels into torch tensor
        data = torch.from_numpy(data_array[i])
        label = torch.from_numpy(label_array[i])
        torch.save({"data": data, "label":label}, path + f'/{imagenumber}_patch{i}.pt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datafile', '-d', required=True,
            help='CZI data hdf5 file containing all czi images')
    parser.add_argument('--mask', '-m', required=True,
            help='Hdf5 file for all czi masks')
    parser.add_argument('--image', '-i', required=True,
            help='czi image file number i.e 1327,1900,2255 etc.')
    parser.add_argument('--msifile', required=True,
            help='czi image file number i.e 1327,1900,2255 etc.')
    parser.add_argument('--patchsize', type=int, default=512,
            help='Size of patches')
    parser.add_argument('--path', '-p', required=True,
            help='path to save the patches')
    args = parser.parse_args()

    P = args.patchsize

    # load the image data from hdf5 file
    with h5py.File(args.datafile,'r') as f1:
        image = f1[f'features_{args.image}']
        img = np.array(image)

    # load corresponding mask
    with h5py.File(args.mask,'r') as f2:
        mask = f2[f'labels_{args.image}']
        label = np.array(mask)

    with h5py.File(args.msifile, 'r') as msih:
        # this is in CHW where C=200 is num PCA components
        msi = np.array(msih['data'])

    print(label.shape, img.shape, msi.shape)
    assert False

    #for i in range(img.shape

    #create patch for image
    patches_image = patchify(img, (512, 512, 3), step=512)
    patches_image = np.squeeze(patches_image).reshape(patches_image.shape[0]*patches_image.shape[1],512,512,3)
    print(patches_image.shape)

    #create patch for labels
    patches_labels = patchify(label, (512, 512), step=512)
    patches_labels = np.squeeze(patches_labels).reshape(patches_labels.shape[0]*patches_labels.shape[1],512,512)
    print(patches_labels.shape)

    _ = save_patches(patches_image, patches_labels, args.path, args.image)

