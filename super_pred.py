

# This file is aim to predict class label for given Test sample... class from set of super classes

import numpy as np
import pandas as pd
import sklearn as sk
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import requests

def get_Abstract(paperId):

    url="https://www.ncbi.nlm.nih.gov/pubmed/"+str(paperId)
    request = requests.get(url)
    if(request.status_code ==200):
        try:
            page = urlopen(url)
            content = page.read()
            soup = bs(content)
            abstract_tag = soup.findAll('abstracttext')
            abstract_text=''.join([v.string for v in abstract_tag])
            text = abstract_text
        except urllib.error.HTTPError as err:
            print("Page not Exists")
    else:
        text=""
    return text



data=pd.read_csv('Dataset/lung.csv',index_col=0)

PubMed_nr=data['Pubmed_ID']



i=0
abstracts=np.array([])
for k in PubMed_nr:
    print(k)
    text=get_Abstract(k)
    abstracts=np.append(abstracts,text)
    #i=i+1

data['abstract']=abstracts

data.to_csv('lungs_abstract.csv',index=False)

#print(abstracts)

