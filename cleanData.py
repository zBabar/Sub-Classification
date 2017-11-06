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
try:
    from nltk import wordpunct_tokenize
    from nltk.corpus import stopwords
    import nltk
except ImportError:
    sys.stderr.write(pimpString('Error! You need to install nltk (http://nltk.org/index.html)'))
# my library
import textTools

def signal_handler(signal, frame):
    sys.stderr.write(pimpString('You pressed Ctrl+C!','red'))
    sys.exit(0)
    
def pimpString(string, color='', bold=''):
    attr = []
    if not sys.stdout.isatty():
      return string
    if color == 'green' or color == 'done' or 'done' in string.lower():
        # green
        attr.append('32')
    elif color == 'red' or color == 'error' or 'error' in string.lower():
        # red
        attr.append('31')
    elif color == 'orange' or color == 'warning' or 'warning' in string.lower():
      # orange
      attr.append('33')
    elif color == 'blue' or color == 'alt_done':
      # blue
      attr.append('36')
    else:
      # default color white
      attr.append('37')
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def is_uppercase(astr):
    return (len(set(string.ascii_uppercase).intersection(astr.strip()))+len(set(string.digits).intersection(astr.strip()))) == len(astr.strip())
"""
  Class that represent a patient folder with an ID and splitted 
  and cleaned in 3 parts: patient , clinical , pathological
"""

def csv_to_TextFile():
      Data_csv=pd.read_csv('Datasets/all_cancer3.csv',encoding='latin1')
      j=0
      with open("Datasets/Cancer3_10000.txt",'w') as f:
        for i in (Data_csv['text']):
            
            f.write(i)
            j=j+1
            if j==10000:
                break
        f.close()
        
        
def writeFile(Text):
    try:
        with open('Datasets/Cancer_clean.txt', 'w',encoding='utf-8') as content_file:
            content_file.write(Text)
            content_file.close()
                
    except IOError:
        print("Cannot find file",filename)
    
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
      l=self.removeCommon(l)
      
      
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
    
    csv_to_TextFile()
    
    signal.signal(signal.SIGINT, signal_handler)
    #reload(sys)
  #sys.setdefaultencoding("latin-1")
    folders=[]
    csv_Data=[]
    
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
        filename="Datasets/Cancer3_10000.txt"
    
    try:
        with open(filename, 'r',encoding='utf-8') as content_file:
            content = content_file.read()
           
    #print "#Characters:",len(content)
            
            
            current=Folder(content,black_list)
            current.process()
            writeFile(current.clean_text)
            
    except IOError:
        print("Cannot find file",filename)


if __name__ == '__main__':
    
    
    main(sys.argv)
    
    
    

