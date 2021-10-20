import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import hdf5plugin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def logisticRegression(data_list):
    # how to get training data? each pca file is a 3d dataset of shape (200,length of x coordinates, length of y coordinates)


    # how to get labels? 


    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=21)

    # Logistic regression
    clf = LogisticRegression(random_state=21).fit(X_train, y_train)

    #to check the shape of the coefficient matrix
    print(clf.coef_.shape)

    # Evaluate performance
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    auc = roc_auc_score(y_test, y_pred)
    print(f'ROC-AUC: {auc}')

    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', nargs='*', required='True',
            help='Top-level directory of pca files')
    args = parser.parse_args()

    #Load all the h5 datas and save in a list
    datasets = []

    for i in range(6):
        f = h5py.File(args.datadir[i],'r') 
        datasets.append(f['data'])

    _ = logisticRegression(datasets)
