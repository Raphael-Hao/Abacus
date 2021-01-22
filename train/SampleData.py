import numpy as np
import pandas as pd
import random

def getXY(data):
    assert data.shape[1] == 11
    map = {"vgg19" : np.array([1, 0, 0]), "resnet152" : np.array([0, 1, 0]), "inception_v3" : np.array([0, 0, 1])}
    n = data.shape[0]
    X = []
    Y = []
    Z = []
    for i in range(n):
        line = data[i]
        name1 = map[line[0]]
        l1 = np.array([int(line[1])])
        r1 = np.array([int(line[2])])
        bs1 = np.array([int(line[3])])
        name2 = map[line[4]]
        l2 = np.array([int(line[5])])
        r2 = np.array([int(line[6])])
        bs2 = np.array([int(line[7])])
        c = np.concatenate((name1, l1, r1, bs1, name2, l2, r2, bs2))
        X.append(c)
        Y.append(float(line[10]))
        Z.append(float(line[8]) + float(line[9]))
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    return X, Y, Z


def getData(ratio):
    path = "/home/abclzr/data/out_bs12.csv"
    data=pd.read_csv(path, header=None)
    data=data.values.tolist()
    random.shuffle(data)
    n = len(data)
    print(n)
    data = np.array(data)
    X_train = data[:int(n*ratio),:]
    X_test = data[int(n*ratio):,:]
    
    assert X_train.shape[1] == 11
    assert X_test.shape[1] == 11
    X_train, Y_train, Z_train= getXY(X_train)
    X_test, Y_test, Z_test = getXY(X_test)
    return X_train, X_test, Y_train, Y_test, Z_train, Z_test

