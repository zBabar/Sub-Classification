


from ibmdbpy.feature_selection import info_gain
from ibmdbpy import IdaDataFrame
from ibmdbpy.base import IdaDataBase

import heapq
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif


def apply_Model(X, Y):
    # X_train=X_train.iloc[:,0:1139]
    idadb = IdaDataBase("DASHDB")
    idadf = IdaDataFrame(idadb)
    IG=info_gain(idadb,target=Y,features=X)


def import_Data():
    Data = pd.read_csv('Disease_Data1.csv')
    # print(Data.shape)

    X = Data.iloc[:, 0:1140]

    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_


X, Y, Y_ = import_Data()

IG = apply_Model(X.iloc[:, 0:1139], Y_)


