"""
This script loads a saved hdf5 file, run incremental pca on it and then saved the first 1000 components.
"""

import pyimzml.ImzMLParser as parser
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

def pca(data, n, chunk_size, n_components, batch_size, output):
    '''
    perform pca on the loaded data
    data: dataset to perform PCA on
    n: number of rows in the data - data.shape[0]
    chunk size: how many rows we feed to IPCA at a time, the divisor of n
    n_components: number of PCA componets to compute
    batch_size: batch size for incremental pca. Shoild be less than or equal to chunk size
    '''
    ipca = IncrementalPCA(n_components, batch_size)
    
    for i in tqdm(range(0, n//chunk_size)):
        ipca.partial_fit(data[i*chunk_size : (i+1)*chunk_size])

    h5 = h5py.File(f'{output}-pca-rapiflex.h5', 'w')
    h5.create_dataset('output', shape=(data.shape[0],n_components))

    for i in tqdm(range(0, n//chunk_size)):
        h5['output'][i*chunk_size:(i+1)*chunk_size] = ipca.transform(data[i*chunk_size : (i+1)*chunk_size])
    h5.close()
    print(ipca.components_.shape)


if __name__ == '__main__': 
    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filepath', '-f', required='True',
            help='path of the saved hdf5 file')
    parser.add_argument('--output', '-o', required='True',
            help='path for the output file')
    args = parser.parse_args()

    f = h5py.File(args.filepath,'r') 
    data = f['data']
    pca_data = data[0:data.shape[0]:100].reshape(data.shape[0]//100, data.shape[1]*data.shape[2])
    pca_data = pca_data.T
    f.close()
    
    pca(pca_data,pca_data.shape[0],1000,1000,512,args.output)

    out = h5py.File(f'{args.output}-pca-rapiflex.h5','r') 
    pca_dataset = out['output']
    print(pca_dataset[:,1].shape)



