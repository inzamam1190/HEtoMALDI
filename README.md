# Agar Data Analysis

## imzML to hdf5
Use spectra.py to convert an imzML file to hdf5 file. You can run it with two required arguments - path of the imzML file and path to save the hdf5 file.

`python spectra.py -d path-of-imzML-file -f path-to-save-hdf5-file`

You can then load the saved hdf5 file as:
```
f = h5py.File(path-to-save-hdf5-file,'r') 
data = f['data']
```
## PCA
Use pca_spectra.py to load a hdf5 file saved on disk and run incremental PCA and save the first 1000 components. You can run it with two required arguments - path of the saved the hdf5, and path to save the PCA output.

`python pca_spectra.py -f path-of-saved-hdf5-file -o path-to-save-PCA-output`

You can then load the PCA output file as:
```
f = h5py.File(path-to-save-PCA-output,'r') 
output = f['output']
```
