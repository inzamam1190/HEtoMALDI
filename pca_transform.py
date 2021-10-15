import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import hdf5plugin
import pickle as pk


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', required='True',
            help='Top-level directory of hdf5 file to transform')
    parser.add_argument('--filename', '-f', required='True',
            help='name of the saved pca file')
    parser.add_argument('--model', '-m', required='True',
            help='Saved trained PCA model')
    args = parser.parse_args()

    with open(args.model, 'rb') as pickle_file:
    ipca = pk.load(pickle_file)

    f = h5py.File(args.datadir,'r') 
    intensity_data = f['interpolated_intensities']

    with h5py.File(args.filename,'w') as f:
        pca = f.create_dataset('data', shape=(ipca.components_.shape[0],intensity_data.shape[1],intensity_data.shape[2]), 
                               chunks=(ipca.components_.shape[0],1,1), dtype='f')

        for i in tqdm(range(pca.shape[1])):
            for j in range(pca.shape[2]):
                pca[:, i, j] = ipca.transform(np.array([intensity_data[:, i, j]]))

    f.close()