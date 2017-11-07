#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:13 2017

@author: zaheerbabar
"""
import sys, traceback
import signal
import re
import os
import random
import shutil
import string
import time
import datetime
from pprint import pprint
from math import log10
from imp import reload
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



def DocsToFMatrix(docs):
    # initialize the  vectorizer
    vectorizer = CountVectorizer(min_df=20)
    x1 = vectorizer.fit_transform(docs)
    # create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index=vectorizer.get_feature_names())
    return df



def main(argv):

    clean_Disease=pd.read_csv('Dataset/clean_disease.csv')
    Tm = DocsToFMatrix(clean_Disease['clean_abstract'])
    Tm = Tm.transpose()

    Tm['Class'] = clean_Disease['Class']
    print(Tm)
    Tm.to_csv('Disease_Data1.csv', index=False)


if __name__ == '__main__':
    main(sys.argv)




