
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


## Data import and seperate

def import_Data():
    Data=pd.read_csv('Disease_Data.csv')
    print(Data.shape)

    X=Data.iloc[:,0:1141]


    Y=Data['Class']
    Y_=Data['Subject']

    return X,Y,Data

## spliting into test and training data

def split_class_Data(Data,Y):
    Data['Class']=Y
    sub_class={}
    super_classes=list(set(Y))
    for cls in super_classes:
        #print(cls)
        sub=Data[Data['Class']==cls]
        sub_class[cls]=sub
    return sub_class



def split_Data(X,Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


    return X_train, X_test, y_train, y_test


def apply_Model(X_train,y_train):
    X_train=X_train.iloc[:,0:1140]
    y_train=pd.DataFrame(y_train)
    print(type(y_train))


    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    return clf

def apply_sub_Model(sub_class,Y):
    sub_Models={}
    for cls in list(set(Y)):
        Data=sub_class[cls]
        X_train=Data.iloc[:,0:1141]
        y_train=Data['Subject']
        model=apply_Model(X_train,y_train)
        sub_Models[cls]=model
    return sub_Models


def model_Predict(clf,X_test):

    y_pred=clf.predict(X_test.iloc[:,0:1140])
    #acc = accuracy_score(y_test, y_pred)
    return y_pred

def super_Predict(X,Y):
    X_train, X_test, y_train, y_test = split_Data(X, Y)

    model = apply_Model(X_train, y_train)
    y_pred = model_Predict(model, X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred,X_train,X_test,y_train,y_test

def sub_predict(y_pred, X_train, X_test, y_train, y_test,Y):
    y_sub_pred=np.array([])
    sub_class = split_class_Data(X_train,Y)
    #print(sub_class)
    sub_Models=apply_sub_Model(sub_class,Y)
    #print(sub_Models)



def main():
    #super class processing and prediction
    X,Y,Data=import_Data()
    y_pred, X_train, X_test, y_train, y_test=super_Predict(X,Y)

    #print(X_train)

    # sub class prediction
    sub_predict(y_pred, X_train, X_test, y_train, y_test,Y)

#clf.score(X_test, y_test)

#print(result)
main()