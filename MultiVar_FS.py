
import pandas as pd
import numpy as np
import pymrmr




def apply_Model(X,Y):

    result=pymrmr.mRMR(X,'MIQ',10)
    print(result)

def import_Data():
    Data = pd.read_csv('Disease_Data1.csv')
    # print(Data.shape)

    X = Data.iloc[:, 0:Data.shape[1] - 2]

    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_



X,Y,Y_=import_Data()

#model = apply_Model(X,Y_)


