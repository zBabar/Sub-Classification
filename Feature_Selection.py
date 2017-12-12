import heapq
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)


def apply_Model(X, Y):

    # Chi2 based feature selection
    CH=SelectKBest(chi2, k=50).fit(X, Y)
    Feature_CH = X.iloc[:, CH.get_support(True)].columns
    #h=heapq.nlargest(10, range(len(Sub.scores_)), Sub.scores_.take)

    #Mutual Information based feature selection
    MI = SelectKBest(mutual_info_classif, k=50).fit(X, Y)
    Feature_MI= X.iloc[:, MI.get_support(True)].columns

    #Information Gain based feature selection

    DT = DecisionTreeClassifier(criterion='entropy',random_state=0)
    DT=DT.fit(X,Y)
    h=heapq.nlargest(50, range(len(DT.feature_importances_)), DT.feature_importances_.take)
    Feature_IG = X.columns[h]
    #print(h,MI.get_support(True),CH.get_support(True))

    # Logistic Regression based feature selection
    LR= linear_model.LogisticRegression(C=1e5).fit(X, Y)
    h=LR.coef_
    h = np.mean(h, axis=0)
    print(h.shape)
    h = heapq.nlargest(50, range(len(h)), h.take)
    Feature_LR = X.columns[h]





    T=np.intersect1d(Feature_CH,Feature_MI)
    #print(np.intersect1d(T,Feature_IG))
    return Feature_CH,Feature_MI,Feature_IG,Feature_LR

def import_Data():
    Data = pd.read_csv('Disease_Data1.csv')
    NumOfFeatures = Data.shape[1]-2

    X = Data.iloc[:, 0:NumOfFeatures]



    Y = Data['Class']
    Y_ = Data['Subject']

    return X, Y, Y_


X, Y, Y_= import_Data()

Ch2,MI,IG,LR = apply_Model(X, Y_)


# Ch2=np.sort(Ch2)
# MI=np.sort(MI)
# IG=np.sort(IG)
# LR=np.sort(LR)




SelectWords={}

SelectWords['MI']=MI
SelectWords['Ch2']=Ch2
SelectWords['IG']=IG
SelectWords['LR']=LR

#words=list(set(MI)|set(IG)|set(LR)|set(Ch2))

#Temp=pd.DataFrame([],index=words)

SelectWords=pd.DataFrame(SelectWords)

print(SelectWords)

SelectWords.to_csv('Selected_Features.csv')

#result=pd.concat([MI_df,Ch2_df],axis=1)

# MI_df=pd.DataFrame(MI,index=MI)
# Ch2_df=pd.DataFrame(Ch2,index=Ch2)
# IG_df=pd.DataFrame(IG,index=IG)
# LR_df=pd.DataFrame(LR,index=LR)
#
# Temp['MI']=MI_df
# Temp['Ch2']=Ch2_df
# Temp['IG']=IG_df
# Temp['LR']=LR_df
#
# #print(Temp.shape)
#
# Temp[Temp.notnull()] = 1
#
# Temp[Temp.isnull()] = 0
#
# print(Temp.shape)

