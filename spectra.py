"""
This script converts an imzML file to a hdf5 file. If you want to see the metadta of the imzML file, see how to do that in msi.ipynb file. The converted hdf5 file will be of shape (len(coordinates), max length of x coordinate, max length of y coordinate).
"""

import pyimzml.ImzMLParser as parser
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

def geth5(dir,filename):
    """
    Convert imzML file to hdf5 file
    """

    # Load the imzML image
    p = ImzMLParser(dir)
    # Metadtata
    print(p.imzmldict)
    #create hdf5 dataset and write to it
    mzA, intA = p.getspectrum(10)
    with h5py.File(f'{filename}-rapiflex.h5', 'w') as f:
        im = f.create_dataset('data', shape=(len(intA),p.imzmldict['max count of pixels x']+1,p.imzmldict['max count of pixels y']+1), chunks=(len(intA), 1,1), dtype='f')
        for i,(x,y,z) in tqdm(enumerate(p.coordinates),total=len(p.coordinates)):
            _, intensity = p.getspectrum(i)
            im[:,x,y] = np.asarray(intensity)
        f.close()
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', required='True',
            help='Top-level directory of pyimzML file')
    parser.add_argument('--filename', '-f', required='True',
            help='name of the saved file')
    args = parser.parse_args()

    geth5(args.datadir, args.filename)
