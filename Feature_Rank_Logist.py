
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB



def apply_Model(X,Y):
    #X_train=X_train.iloc[:,0:1139]


    clf=linear_model.LogisticRegression(C=1e5).fit(X, Y)
    feature_weight=pd.DataFrame(clf.coef_,columns=X.columns,index=clf.classes_)
    feature_word_relv=feature_weight.idxmax(axis=0)
    feature_word_relvSort=feature_word_relv.sort_values(ascending=True)
    print(type(feature_word_relvSort))
    feature_word_relvSort.to_csv("feature_wordRelv.csv")
    #print(feature_word_relv['hormone'])


def import_Data():
    Data = pd.read_csv('Disease_Data1.csv')
    # print(Data.shape)

    X = Data.iloc[:, 0:Data.shape[1] - 2]

    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_



X,Y,Y_=import_Data()

model = apply_Model(X,Y_)


