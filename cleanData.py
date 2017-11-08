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


try:
    from nltk import wordpunct_tokenize
    from nltk.corpus import stopwords
    import nltk
except ImportError:
    sys.stderr.write('Error! You need to install nltk (http://nltk.org/index.html)')
# my library
import textTools

def signal_handler(signal, frame):
    sys.stderr.write(pimpString('You pressed Ctrl+C!','red'))
    sys.exit(0)


def is_uppercase(astr):
    return (len(set(string.ascii_uppercase).intersection(astr.strip()))+len(set(string.digits).intersection(astr.strip()))) == len(astr.strip())
"""
  Class that represent a patient folder with an ID and splitted 
  and cleaned in 3 parts: patient , clinical , pathological
"""
def DocsToFMatrix(docs):
    # initialize the  vectorizer
    vectorizer = CountVectorizer(min_df=15)
    x1 = vectorizer.fit_transform(docs)
    # create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index=vectorizer.get_feature_names())
    return df

class Folder:
  
  
  def __init__(self,plain_text,black_list=[],debug=False):
    """
      Instantiate the class Folder starting from the folder plain text
      @param plain_text as the variable name suggest, it represents there plain folder text
      @type text: str
      
    """
    self.debug=debug
    self.plain_text=plain_text
    self.black_list=black_list
    
    self.unknown_initials=[]
    
  def process(self):
    """
      Process the plain text and then split it in the correct variables
    """
    self.semi_clean_text=self.clean(self.plain_text)
    #print self.patient_text
    
    self.clean_text=textTools.doStem(nltk.word_tokenize(self.semi_clean_text))  
    #print self.clinical_text
    
  def clean(self,text,onlyAlphaNum=True):
    """
      Clean the input from the symbols that are not letters and from unwanted string line
      @param text the text to clean
      @type text: str
      @param onlyAlphaNum True if you want to leave only the letters and numbers in the text
      @type onlyAlphaNum bool
      
      @return clean_lines the cleaned text
      @rtype clean_lines: str
    """
    clean_lines=""
    lines=text.split("\n") 
    #print("After splitting", lines)
    first_line=True
    for l in lines:
      l=l.strip()
      #l=self.removeDates(l)  
      #edssL=self.isEdssLine(l)
      #edssL=""
      #if edssL:
	#l=edssL
      #elif onlyAlphaNum:
      if(onlyAlphaNum):
          l=''.join(e for e in l if e.isalnum() or e==" ")
      l=l.strip()
      if not l or not l.find("Page") or not l.find("Neuropathological code") or not l.find("Medication"):
          continue
      
      #print("Lets Check",l)
      #l=self.removeCommon(l)
      
      
      #print("Lets Check again",l)
      #l=self.initialsToWord(l)
      
      if first_line:
          clean_lines=clean_lines + l
          first_line=False
      else:
          clean_lines=clean_lines + " " + l
    
    return clean_lines

  def removeDates(self,astr):
    """
      Remove the dates from the text
      @param astr the string to analyzed
      @type text: str
      
      @return clean_str the cleaned string
      @rtype clean_str: str
    """
    clean_str=""
    astrs=re.split(" +",astr)
    first_line=True
    for w in astrs:
      #print "Length of splitting:",len(re.split("/|-",w)),"for word",w
      if (len(re.split("/|-",w))>1 and re.split("/|-",w)[0].isdigit() and re.split("/|-",w)[1].isdigit()) or not w :
          continue
      if first_line:
          clean_str=clean_str + w
          first_line=False
      else:
          clean_str=clean_str + " " + w
    return clean_str

  def removeCommon(self,line):
    """
      Remove the common unwanted string from a lines
      @param line the line to analyzed
      @type text: str
      
      @return clean_str the cleaned string
      @type clean_str: str
    """
    common_strings=["Cause of death"]
    clean_str=""  
    for c in common_strings:
      position=line.lower().find(c.lower())   
      if not position==-1:
          clean_str= line[(position+len(c)):len(line)]
      else:
          clean_str=line
    return clean_str


  def isEdssLine(self,line):
    ret_line=''
    if 'edss' in line.lower():
      strEdss=line.split(" ")
      if not len(strEdss)<3:
          ret_line='edss' + strEdss[1]
    return ret_line


  def initialsToWord(self,line):
    wellknown_initials=["F","M","MS"]
    wellknown_initials_words=["Female","Male","Multiple Sclerosis"]
    ret_line=""
    first_word=True
    
    for w in line.split(" "):
      word=w.strip()
      if not word:
          continue
      if is_uppercase(word):
          try:
              index_word=wellknown_initials.index(word)
              word=wellknown_initials_words[index_word]
          except ValueError:
              if not word in self.unknown_initials:
                  self.unknown_initials.append(word)
                  continue
      if first_word:
            	ret_line=word
            	first_word=False
      else:
          ret_line=ret_line + " " + word

    return ret_line
  def _calculate_languages_ratios(self,text):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}
    
    @param text: Text whose language want to be detected
    @type text: str
    
    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {}

    '''
    nltk.wordpunct_tokenize() splits all punctuations into separate tokens
    
    >>> wordpunct_tokenize("That's thirty minutes away. I'll be there in ten.")
    ['That', "'", 's', 'thirty', 'minutes', 'away', '.', 'I', "'", 'll', 'be', 'there', 'in', 'ten', '.']
    '''

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    
    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios


  #----------------------------------------------------------------------
  

  
  
def main(argv):

    clean_abstract=np.array([])
    black_list=['show',
	      'year',
	      'many',
	      'due'#,
	      #'eds'
	      #'type',
	      #'reveal',
	      #'serie',
	      #'column',
	      #'define',
	      #'string',
	      #'contains',
	      #'till',
	      #'patient',
	      #'give',
	      #'notice'
	      ]
    if len(argv)>1:
        filename=argv[1]
    else:
        filename="Dataset/All_Diseases.csv"
    
    try:
        Disease_Data=pd.read_csv(filename)
        abstracts=Disease_Data['abstract']
        abstracts=np.array(abstracts)
        i=0
        for content in abstracts:

            content=str(content)
            if(not content.strip()):
                clean_abstract = np.append(clean_abstract, "")
            else:
                #print(i, content)
                current = Folder(content, black_list)
                current.process()

                clean_abstract = np.append(clean_abstract, current.clean_text)

            i=i+1
        # writeFile(current.clean_text)
        Disease_Data['clean_abstract']=clean_abstract
        Disease_Data.to_csv('Dataset/clean_disease.csv',encoding='utf-8',index=False)
        # Tm=DocsToFMatrix(clean_abstract)
        # Tm=Tm.transpose()
        #
        # Tm['Class']=Disease_Data['Class']
        # print(Tm)
        # Tm.to_csv('Disease_Data.csv',index=False)
    except IOError:
        print("Cannot find file",filename)
    # content='Epidemiologic studies of acute myocardial infarction(AMI) have described gender differences in the time of death after infarction, with greater numbers of men dying before hospitalization than women.However, in controlled, hospital-based clinical trials, women die at higher rates than men.We hypothesized that evidence of a gender difference in the time of death following AMI may be found in controlled studies of hospitalized AMI patients.We performed a retrospective analysis of the Global Utilization of Streptokinase and Tissue Plasminogen Activator for Occluded Coronary Arteries (GUSTO-1) and International Joint Efficacy Comparison of Thrombolytics (INJECT) trial databases using logistic regression modeling and time-to-death analyses.The age-adjusted female-to-male odds ratio for mortality was 1.4 (95 % confidence interval 1.3 to 1.5) in GUSTO-1 and 1.5 (95 % confidence interval 1.3 to 1.8) in INJECT.GUSTO-1 showed that among patients dying during the first 24 hours after symptom onset, men died an average of 1.7 hours earlier than women (p < 0.001).This difference was due to earlier deaths among men < or =65 years of age.Furthermore, in GUSTO-1, the analysis of time to death in hour increments demonstrated that greater proportions of men died at earlier time points than women and a disproportionate number of early deaths occurred among younger men than among women of any age or older men.In INJECT, where time to death could only be analyzed in 1-day increments, no gender differences were evident.These results raise the possibility that the pattern of earlier death for men in thrombolytic clinical trials represents the continuation of a gender-specific mortality pattern that began before hospitalization.The death of a disproportionate number of men before hospitalization may represent an inherent gender bias for clinical studies enrolling only hospitalized patients.More high-risk men would be excluded from these studies than women because of death before hospitalization.Hence, gender comparisons of in -hospital mortality rates may artificially inflate values for women.'
    # current = Folder(content, [])
    # current.process()
    # print(current.clean_text)


if __name__ == '__main__':
    
    
    main(sys.argv)
    
    
    

