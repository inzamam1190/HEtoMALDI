import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as scnd
#import skimage
from scipy import interpolate as scinterp
import h5py
from tqdm import tqdm
import hdf5plugin
import pyimzml.ImzMLParser as parser
from pyimzml.ImzMLParser import ImzMLParser


def interpolation_msi(p, min_mz:int, max_mz:int, interpStep:float, filename:str):
    '''
    interpolates the data to same m/z values and convert to hdf5 file.

    p = imzML parsed object
    min_mz = Minimum m/z value to start interpolation from
    max_mz = MAximum m/z value to end interpolation
    interpStep = interpolation step
    filename = hdf5 filename to save to disk
    '''
    
    mz, _ = p.getspectrum(0) #Load a random spectrum to get the m/z dtype
    
    common_mz = (np.arange(start=int(min_mz), stop=int(max_mz), step=interpStep)).astype(mz.dtype) #common m/z values for interpolating the data
    
    f = h5py.File(filename, 'w')
    mz_values = f.create_dataset('common_mz', shape=common_mz.shape, dtype=common_mz.dtype) #dataset for the common m/z

    #dataset for the intensity values
    intensity_values = f.create_dataset('interpolated_intensities', shape=(len(common_mz), 
                            int(1+p.imzmldict['max count of pixels x']), int(1+p.imzmldict['max count of pixels y'])), 
                                        chunks=(common_mz.shape[0], 1,1), dtype='f')
    
    
    for i,(x,y,z) in tqdm(enumerate(p.coordinates),total=len(p.coordinates)):
        m_by_z, intensity = p.getspectrum(i)
        itest = scinterp.interp1d(m_by_z, intensity, kind="slinear", fill_value="extrapolate")
        interped = itest(common_mz)
        intensity_values[:,x,y] = interped
    
    mz_values[:] = common_mz
    
    f.close()
        
    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', required='True',
            help='Top-level directory of pyimzML file')
    parser.add_argument('--filename', '-f', required='True',
            help='name of the saved file')
    args = parser.parse_args()

    p = ImzMLParser(args.datadir) #parse the imzML file

    print(p.imzmldict) #print the metadata

    _ = interpolation_msi(p, 50, 2800, 0.001, args.filename)

    f = h5py.File(args.filename,'r') 
    intensity_data = f['interpolated_intensities']
    common_mz = f['common_mz']

    print(intensity_data)
    print(common_mz)

    mz, inte = p.getspectrum(90)

    plt.figure(figsize=(16, 8))
    plt.plot(mz, inte, linewidth=5, label="Original Data")
    plt.plot(common_mz[:], intensity_data[:,p.coordinates[90][0],p.coordinates[90][1]], label="Interpolated Data")
    plt.xlim(100,1200)
    plt.ylim(0)
    plt.xlabel("m/z values")
    plt.legend(loc="upper right")
    plt.show()

    f.close()