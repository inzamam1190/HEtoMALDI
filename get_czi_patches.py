import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from tqdm import tqdm
import hdf5plugin
from patchify import patchify

def save_patches(data_array, mask_array, path:str):
    for i in tqdm(range(data_array.shape[0])):  
        # convert image patches into torch tensor
        data = torch.from_numpy(data_array[i])

        # threshold mask array and get label for the patch
        if mask_array[i].mean() >= 0.5:
            label = 1
        else: 
            label = 0
        # save data and label in the same .pt file as dictionaries
        torch.save({"data": data, "label":label}, path + f'/patch{i}.pt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datafile', '-d', nargs='*', required='True',
            help='CZI data hdf5 file containing all czi images')
    parser.add_argument('--mask', '-m', nargs='*', required='True',
            help='Hdf5 file for all czi masks')
    parser.add_argument('--image', '-i', nargs='*', required='True',
            help='czi image file number i.e 1327,1900,2255 etc.')
    parser.add_argument('--path', '-p', nargs='*', required='True',
            help='path to save the patches')
    args = parser.parse_args()

    # load the image data from hdf5 file
    f1 = h5py.File(args.datadir,'r')
	image = f1[f'features_{args.image}']

	# load corresponding mask
	f2 = h5py.File(args.mask,'r')
	mask = f2[f'labels_{args.image}']

	print(image)
	print(label)

	img = np.array(image)
	label = np.array(mask)

	#create patch for image
	patches_image = patchify(img, (512, 512, 3), step=512)
	patches_image = np.squeeze(patches_image).reshape(patches_image.shape[0]*patches_image.shape[1],512,512,3)
	print(patches_image.shape)

	#create patch for labels
	patches_labels = patchify(label, (512, 512), step=512)
	patches_labels = np.squeeze(patches_labels).reshape(patches_labels.shape[0]*patches_labels.shape[1],512,512)
	print(patches_labels.shape)

	del img, label

	_ = save_patches(patches_image, patches_labels, args.path)

	f1.close()
	f2.close()

