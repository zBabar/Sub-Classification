
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB



def apply_Model(X,Y):
    #X_train=X_train.iloc[:,0:1139]
    feature_word={}

    clf=linear_model.LogisticRegression(C=1e5).fit(X, Y)
    feature_weight=pd.DataFrame(clf.coef_,columns=X.columns,index=clf.classes_)
    feature_word_relv=feature_weight.idxmax(axis=0)
    #print(feature_weight)
    #feature_word_relvSort=feature_word_relv.sort_values(ascending=True)
    fwr=pd.DataFrame(feature_word_relv)
    fwr['score']=feature_weight.max()
    #print(fwr)
    for i in set(Y):
        #print(i)
        word_score={}
        #word_score['words']=fwr[fwr[0]==i].index
        #word_score['score']=fwr[fwr[0] == i]['score']
        #feature_word[i]=word_score
        #feature_word[i]=(fwr[fwr[0] == i]['score'].sort_values(ascending=False))[:10]
        word_score['words']=np.array((fwr[fwr[0] == i]['score'].sort_values(ascending=False))[:10].index)
        word_score['score'] = np.array((fwr[fwr[0] == i]['score'].sort_values(ascending=False))[:10].values)
        feature_word[i]=word_score
    feature_word=pd.DataFrame(feature_word)
    feature_word.to_csv('feature_rank_classwise.csv')
    #print(feature_word_relv['hormone'])



def import_Data():
    Data = pd.read_csv('Disease_Data1.csv')
    # print(Data.shape)

    X = Data.iloc[:, 0:1140]

    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_



X,Y,Y_=import_Data()

model = apply_Model(X.iloc[:,0:1139],Y_)


