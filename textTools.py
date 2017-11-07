import sys

try:
  from nltk import wordpunct_tokenize
  from nltk.stem.wordnet import WordNetLemmatizer
  from nltk.corpus import stopwords
  from nltk.corpus import wordnet
  import nltk
except ImportError:
  print ('[!] You need to install nltk (http://nltk.org/index.html)')
def get_wordnet_pos(treebank_tag):
  
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return ''
  # TO TEST
def doStem(words):
  lemmatizer = WordNetLemmatizer()
  tags=nltk.pos_tag(words)
  i=0
  ret_str=""
  first_word=True
  #print(words)
  for w in words:
    post_tag=get_wordnet_pos(tags[i][1])
    #print post_tag
    try:
      if not post_tag or w.lower().strip() in stopwords.words('english') or len(w)<3:
	#if w.lower().strip() in stopwords.words('english'):
	  #print "STOPWORD:",w.strip()
          continue
      word=lemmatizer.lemmatize(w.lower().strip(), post_tag)
      if first_word:
          first_word=False
          ret_str=word
      else:
          ret_str=ret_str+" "+word
          #print("total string for this iter: ",ret_str )
    finally:
      i=i+1
  return ret_str
def isAWasteWord(word='',idf=0.0,black_list=[],idf_threshold_min=0.0,idf_threshold_max=100.0):
  return (idf<idf_threshold_min or idf>idf_threshold_max or word.strip() in black_list)
    
def main(argv):
  print (doStem(["brought"]))
if __name__ == '__main__':
  main(sys.argv)