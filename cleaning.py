# -*- coding: utf-8 -*-
import pandas as pd
import re
import os
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def cleaning_data(data):
  
    data = [re.sub(r'[A-Za-z0-9\/,@:&\.\(\)\[\]?!\#\_+\|\:]+', ' ', x) for x in data]
    data = ' '.join(data).split()
    
    lines2 = []
    for word in data:
        if word != " " :
            lines2.append(word)

    #stemming 
    print("stemming")
    s_stemmer = SnowballStemmer(language='arabic')
    stemmed = []
    for word in lines2:
        stemmed.append(s_stemmer.stem(word))
    cleaned = s_stemmer.stem(stemmed).set(stopwords)
    #cleaned = [word for word in stemmed if word not in stopwords.words('arabic')]
    
    return cleaned
    
    
if __name__ == '__main__':
    all_sentences =[]
    data = pd.read_json("active_words_ar.json",lines = True,encoding ='utf8')
    df = data["tweet"]
    for sentence in df:
        all_sentences.append(sentence)
        
    #split into words
    lines = list()
    for line in all_sentences:    
        words = line.split()
        for w in words: 
            lines.append(w)
    cleaned_data =  cleaning_data(lines)