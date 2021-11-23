import numpy as np
import czifile
import h5py
from tqdm import tqdm
import hdf5plugin

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filepath', '-f', required='True',
            help='czi filepath')
    parser.add_argument('--target_file', '-t', nargs=2, required='True',
            help="HDF5 filename followed by dataset name to write. Must not exist.")
    args = parser.parse_args()


    target_file, target_dataset = args.target_file

    im = czifile.imread(args.filepath)
    im = im.squeeze() 
    print(f'czi_shape = {im.shape}')

    #Remove background
    im[im > 2000] = 0

    #change shape to CHW
    im = im.transpose(2, 1, 0)

    #write to hdf5
    with h5py.File(target_file, "a") as ftarget:
        if target_dataset in ftarget:
                raise RuntimeError(
                    f"Refusing to overwriting existing dataset {target_dataset} in {target_file}"
                )
        ds_target = ftarget.create_dataset(
                    target_dataset, shape=im.shape, chunks=True, dtype='f', data=im
                ) 

    #check if write was successfull
    
    f = h5py.File(target_file, "r")
    image = f[target_dataset]
    print(image) 

    f.close()