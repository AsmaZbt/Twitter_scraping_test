import pandas as pd
import re
import string
import nest_asyncio
import twint
from nltk.corpus import stopwords
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
nest_asyncio.apply()

c = twint.Config()
#c.Username = "afaaa73"
c.Search ="عقار OR استثمار OR رسول OR منتجات OR مقاطعة OR اللقاح OR كورونا"
c.Lang = "ar"
c.Store_json= True
c.Output = "active_words_ar.json"

output = twint.run.Search(c)

data = pd.read_json("active_words_ar.json",lines = True,encoding ='utf8')

data.shape
data.head()

data.to_csv("arabic_tweets.csv",encoding="utf8")

# cleaning data
dt = data["tweet"]
#print(dt)

#split to words
all_sentences = []

for word in dt:
    all_sentences.append(word)

lines = list()
for line in all_sentences:    
    words = line.split()
    for w in words: 
       lines.append(w)

print(lines)

#Remove punctuation,special caracter, non arabic letters 

lines = [re.sub(r'[A-Za-z0-9]+', ' ', x) for x in lines]
lines = [re.sub(r'[/,@:&.\(\)\[\]?؟!\#\_+]+', ' ', x) for x in lines]
lines = ' '.join(lines).split()

lines2 = []

for word in lines:
    if word != " " :
        lines2.append(word)
print(lines)

#stemming step

from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='arabic')

stemmed = []
for word in lines2:
    stemmed.append(s_stemmer.stem(word))
    
stemmed

# remove stop words


stem = [word for word in stemmed if word not in stopwords.words('arabic')]

dt = pd.DataFrame(stem)

dt = dt[0].value_counts()

# get word frequency
from nltk.probability import FreqDist

freq_words = FreqDist()

for words in dt:
    freq_words[words] += 1

freq_words

# get the top words on the tweets
dt = dt[:50,]
#data = arabic_reshaper.reshape(dt) # check again
plt.figure(figsize=(10,5))
sns.barplot(dt.values, dt.index, alpha=0.8)
plt.title('Top Words')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()




