





import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import sys


sys.setdefaultencoding("ISO-8859-1")

Data=pd.read_csv('Disease_Data.csv',encoding='ISO-8859-1')
print(Data.shape)
# X_train, X_test, y_train, y_test = train_test_split(
#     , iris.target, test_size=0.4, random_state=0)


