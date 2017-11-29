import heapq
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif


def apply_Model(X, Y):
    # X_train=X_train.iloc[:,0:1139]

    CH=SelectKBest(chi2, k=50).fit(X, Y)
    #h=heapq.nlargest(10, range(len(Sub.scores_)), Sub.scores_.take)

    MI = SelectKBest(mutual_info_classif, k=50).fit(X, Y)

    Feature_CH=X.iloc[:,CH.get_support(True)].columns

    Feature_MI= X.iloc[:, MI.get_support(True)].columns

    #print(np.intersect1d(Feature_CH,Feature_MI))
    return Feature_CH,Feature_MI

def import_Data():
    Data = pd.read_csv('Disease_Data1.csv')
    # print(Data.shape)

    X = Data.iloc[:, 0:1140]

    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_


X, Y, Y_ = import_Data()

Ch2,MI = apply_Model(X.iloc[:, 0:1139], Y_)


