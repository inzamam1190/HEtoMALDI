import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle as pk
import matplotlib.cbook as cbook
from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', '-f', required='True',
            help='File directory for which to compute the peak prediction')
    parser.add_argument('--model', '-m', required='True',
            help='Saved trained PCA model')
    parser.add_argument('--transform', '-t', required='True',
            help='filepath for the transformed HDF5 file')
    args = parser.parse_args()

    with open(args.model, 'rb') as pickle_file:
        ipca = pk.load(pickle_file)

    print(f"PCA model component shape: {ipca.components_.shape}")

    heToMsi_file = np.load(args.file)
    print(f"He to MSI file shape: {heToMsi_file.shape}")

    with h5py.File(args.transform,'w') as f:
        reverse = f.create_dataset('data', shape=(900000, heToMsi_file.shape[1], heToMsi_file.shape[2]), 
                               chunks=(900000,1,1), dtype='f')

        for i in tqdm(range(reverse.shape[1])):
            for j in range(reverse.shape[2]):
                reverse[:, i, j] = ipca.inverse_transform(np.array([heToMsi_file[:, i, j]]))

    f = h5py.File(args.transform, 'r')
    data = f['data']
    print(data.shape)

    assert data.shape == (900000, heToMsi_file.shape[1], heToMsi_file.shape[2])


