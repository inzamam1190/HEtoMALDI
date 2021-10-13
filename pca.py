import numpy as np
import matplotlib.pyplot as plt
import stemtool as st
import h5py
from tqdm import tqdm
import hdf5plugin
from sklearn.decomposition import IncrementalPCA


def pca(n_components, batch_size, xmin, xmax, ymin, ymax, intensity_data):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    #take x-coordinates values of the region for training
    xs = np.arange(xmin, xmax,1)
    #take y-coordinates values of the region for training
    ys = np.arange(ymin, ymax,1)

    # Make tuples with xs,ys. Shape must be xs.shape[0]*ys.shape[0]
    tuples = []

    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            tuples.append((xs[i],ys[j]))

    coords = np.array(tuples)
    # shuffle the tuple coordinates for training 
    coords1 = np.random.permutation(coords)
    # make 2d array from 3d intensity data for pca training
    batch = []
    for j in range(coords1.shape[0]):
        batch.append(intensity_data[:, coords1[j][0], coords1[j][1]])
    batch = np.array(batch)

    for i in tqdm(range(0, coords.shape[0]//100)):
        ipca.partial_fit(batch[i*100:(i+1)*100])

    
    with h5py.File(args.filename,'w') as f:
        pca = f.create_dataset('data', shape=(n_components,intensity_data.shape[1],intensity_data.shape[2]), chunks=(n_components,1,1), dtype='f')

        for i in tqdm(range(pca.shape[1])):
            for j in range(pca.shape[2]):
                pca[:, i, j] = ipca.transform(np.array([intensity_data[:, i, j]]))

    return None
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', required='True',
            help='Top-level directory of hdf5 file with interpolated data')
    parser.add_argument('--filename', '-f', required='True',
            help='name of the saved pca file')
    args = parser.parse_args()


    #Load the h5 data
    f = h5py.File(args.datadir,'r') 
    intensity_data = f['interpolated_intensities']
    common_mz = f['common_mz']

    print(intensity_data)

    pca(n_components = 100, batch_size = 100, xmin=100, xmax=125, ymin=10, ymax=50, intensity_data = intensity_data)

    f.close()

    h5 = h5py.File(args.filename,'r')
    pca_data = h5['data']

    print(pca_data.shape[0])

    #plot 1st component
    plt.figure(figsize=(10,6))
    plt.imshow(pca_data[0].T)
    h5.close()



