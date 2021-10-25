import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

def logisticRegression(X,y,k):
    kf = KFold(n_splits=k, shuffle=True, random_state=21)
    model = LogisticRegression(random_state=21, class_weight='balanced', max_iter=5000)
     
    acc_score = []
    AUC_score = []
    F1_score = []
     
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.fit_transform(X_test)
         
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
         
        acc = accuracy_score(y_pred , y_test)
        acc_score.append(acc)
        
        auc = roc_auc_score(y_pred, y_test)
        AUC_score.append(auc)
        
        f1 = f1_score(y_pred, y_test, average='binary')
        F1_score.append(f1)
         
    avg_acc_score = sum(acc_score)/k
    avg_AUC_score = sum(AUC_score)/k
    avg_F1_score = sum(F1_score)/k
     
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    print('Avg AUC : {}'.format(avg_AUC_score))
    print('Avg F1 : {}'.format(avg_F1_score))

    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadirX', '-dX', nargs='*', required='True',
            help='Top-level directory of X(features) numpy files')
    parser.add_argument('--datadiry', '-dy', nargs='*', required='True',
            help='Top-level directory of y(labels) numpy files')
    args = parser.parse_args()

    #Load X and y for different files. X is of shape (N,200) and y is of shape(N,)
    
    X = np.zeros(shape=(0,200))
    for i in range(len(args.datadirX)):
        X = np.concatenate((X,np.load(args.datadirX[i])), axis=0)

    y = np.zeros(shape=(0))
    for i in range(len(args.datadiry)):
        y = np.concatenate((y,np.load(args.datadiry[i])))

    
    _ = logisticRegression(X,y,5)