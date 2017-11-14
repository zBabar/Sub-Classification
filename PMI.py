
import pandas as pd
import numpy as np

df=pd.read_csv('Disease_Data1.csv')

X = df.iloc[:, 0:1139]
#X=df.iloc[:,0:6]


Y = df['Class']
Y_=df['Subject']

X[X>1]=1

#X['Class']=Y
#X['Subject']=Y_
word_class_freq =X.groupby(Y).sum()
whole_sum=np.sum(word_class_freq)
N=np.sum(whole_sum)
word_class_prob=word_class_freq/N

word_freq=np.sum(X)
word_prob=word_freq/len(Y)
class_freq=df['Class'].value_counts()
class_prob=(class_freq/len(Y))[word_class_prob.index]
word_mul_class=pd.DataFrame(np.multiply.outer(class_prob,word_prob))

#
word_mul_class.index=word_class_prob.index

PMI_df=np.array(word_class_prob)/np.array(word_mul_class)

PMI_df[PMI_df==0]=1
PMI=pd.DataFrame(np.log(PMI_df),columns=word_class_freq.columns,index=word_class_freq.index)


print(PMI.idxmax(axis=0))



