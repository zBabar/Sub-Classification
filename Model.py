
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB




## Data import and seperate

def import_Data():
    Data=pd.read_csv('Disease_Data1.csv')
    #print(Data.shape)

    X=Data.iloc[:,0:1140]


    Y=Data['Class']
    Y_=Data['Subject']

    return X,Y,Y_

## spliting into test and training data

def split_class_Data(X,Y): # spliting the data on the bases of classes and storing in a dictionary
    #Data['Class']=Y
    sub_class={}
    super_classes=list(set(Y))
    for cls in super_classes:
        #print(cls)
        sub=X[Y==cls]
        sub_class[cls]=sub
    return sub_class



def split_Data(X,Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


    return X_train, X_test, y_train, y_test


def apply_Model(X_train,y_train):
    #X_train=X_train.iloc[:,0:1139]

    #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    #clf=linear_model.LogisticRegression(C=1e5).fit(X_train, y_train)

    #print(clf.coef_.shape)
    clf = MultinomialNB().fit(X_train, y_train)



    return clf

def apply_sub_Model(sub_class,Y): # training models for data per each class and storing in a dictionary
    sub_Models={}
    for cls in list(set(Y)):
        X=sub_class[cls]
        X_train=X.iloc[:,0:1139]
        y_train=X['Subject']
        model=apply_Model(X_train,y_train)
        sub_Models[cls]=model
    return sub_Models


def model_Predict(clf,X_test):
    #X_test=X_test.iloc[:,0:1139]
    y_pred=clf.predict(X_test)
    #acc = accuracy_score(y_test, y_pred)
    return y_pred

def super_Predict(X,Y): # predicting super class for given test dataset
    X_train, X_test, y_train, y_test = split_Data(X, Y)

    model = apply_Model(X_train.iloc[:,0:1139], y_train)
    y_pred = model_Predict(model, X_test.iloc[:,0:1139])
    acc = accuracy_score(y_test, y_pred)
    print("Super Class Prediction Accuracy:", acc)
    return y_pred,X_train,X_test,y_train,y_test

def sub_predict(y_pred, X_train, X_test, y_train, y_test,Y): # predicting sub class (Subject) for given test data set
    y_sub_pred=np.array([])
    sub_class = split_class_Data(X_train,y_train)
    sub_Models=apply_sub_Model(sub_class,y_train)

    i=0
    for index,case in X_test.iterrows():
        superCls=y_pred[i]
        case=pd.DataFrame(case).iloc[0:1139]
        sub_label=model_Predict(sub_Models[superCls],case.transpose())
        y_sub_pred=np.append(y_sub_pred,sub_label)
        i=i+1
    acc=accuracy_score(X_test['Subject'], y_sub_pred)
    print("Sub Class Prediction Accuracy:", acc)



def main():
    #super class processing and prediction
    X,Y,Y_=import_Data()
    y_pred, X_train, X_test, y_train, y_test=super_Predict(X,Y)

    y_pred1, X_train1, X_test1, y_train1, y_test1 = super_Predict(X, Y_)


    # sub class prediction
    sub_predict(y_pred, X_train, X_test, y_train, y_test,Y)

#clf.score(X_test, y_test)

#print(result)
main()
