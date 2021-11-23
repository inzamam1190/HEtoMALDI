import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import hdf5plugin
import pickle as pk
from sklearn.decomposition import IncrementalPCA


def jointPCA(h5_list):
    ipca = IncrementalPCA(n_components=200, batch_size=100)

    print("Getting training data from h5 datasets......")
    for i in h5_list:
        # For each dataset get every 2nd x,y values
        xs = np.arange(0, i.shape[1], 2)
        ys = np.arange(0, i.shape[2], 2)

        # Make a tuple of x,y values. if we have 20 x-points and 20 y-points there will be 400 tuples.
        tuples = []
        for j in range(xs.shape[0]):
            for k in range(ys.shape[0]):
                tuples.append((xs[j], ys[k]))

        # Shuffle the tuples
        coords = np.random.permutation(
            np.array(tuples)
        )  # shape will be (xs.shape[0]*ys.shape[0],2)

        # empty list for the training data
        train_data = []

        for j in tqdm(range(coords.shape[0])):

            # exclude the background coordinates where there is no spectral signal i.e. sum(intensity) == 0
            if np.sum(i[:, coords[j][0], coords[j][1]]) == 0:
                continue
            # include other coordinates having some signal to the training data
            train_data.append(i[:, coords[j][0], coords[j][1]])
        train_data = np.array(train_data)

        print("Training incremental PCA.....")
        for i in tqdm(range(0, train_data.shape[0] // 200)):
            ipca.partial_fit(train_data[i * 200 : (i + 1) * 200, :])

        del train_data

    with open("joint-ipca.pkl", "wb") as pickle_file:
        pk.dump(ipca, pickle_file)

    print(ipca.explained_variance_ratio_[:10])

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datadir",
        "-d",
        nargs="*",
        required="True",
        help="Top-level directory of hdf5 files with interpolated data",
    )
    args = parser.parse_args()

    # Load all the h5 datas and save in a list
    intensity_datas = []

    for i in range(6):
        f = h5py.File(args.datadir[i], "r")
        intensity_datas.append(f["interpolated_intensities"])

    # run joint PCA
    _ = jointPCA(intensity_datas)
