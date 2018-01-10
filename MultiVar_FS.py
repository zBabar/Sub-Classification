
import pandas as pd
import numpy as np
#import pymrmr
from skfeature.function.information_theoretical_based import JMI, MRMR, MIM, MIFS



def apply_Model(X,Y):

    result=pymrmr.mRMR(X,'MIQ',10)
    print(result)

def import_Data():
    Data = pd.read_csv('Disease_Data_BiGram.csv')
    # print(Data.shape)

    X = Data.iloc[:, 0:Data.shape[1] - 2]

    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_


FS={}
X,Y,Y_=import_Data()

FS['MRMR']=X.columns[MRMR.mrmr(np.array(X),Y_,n_selected_features=15)[:15]]
FS['JMI']=X.columns[JMI.jmi(np.array(X),Y_,n_selected_features=15)[:15]]
FS['MIFS']=X.columns[MIFS.mifs(np.array(X),Y_,n_selected_features=15)[:15]]
FS['MIM']=X.columns[MIM.mim(np.array(X),Y_,n_selected_features=15)[:15]]

FS=pd.DataFrame(FS)
print(FS)
FS.to_csv('Selected_Features_MultiVar_BiG.csv')

#print(pd.DataFrame(FS))
#model = apply_Model(X,Y_)


