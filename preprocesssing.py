# -*- coding: utf-8 -*-
import pandas as pd
import re
import string
import nest_asyncio
nest_asyncio.apply()


other_characters = '''\n\n+`÷×؛<>()*&^%]\[\/:".,'{}~¦+|!”…“–0123456789'''
english_punctuations = string.punctuation
punctuations_list =  english_punctuations + other_characters
 
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, ' ', text)
    return text

def remove_punctuations(text):
    translator = str.maketrans(' ', ' ', punctuations_list)
    return text.translate(translator)

def remove_english_letters(text):
    text = re.sub(r'[a-zA-Z]+', ' ', text)
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def remove_links(text):
    text = re.sub(r'https://[A-Za-z0-9./]+', ' ',text)
    return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', text)

def clean_tweet(tweet):
   tweet = remove_english_letters(tweet)
   tweet = remove_links(tweet)
   tweet = remove_diacritics(tweet)
   tweet = remove_repeating_char(tweet)
   tweet = remove_punctuations(tweet) # need to keep the standard punctuation . ! ? 
   tweet = remove_emoji(tweet)   #we will look later to convert emoji to words to improve the accuracy
   tweet = re.sub(r" +"," ",tweet) 
    # may be remove stop word here
  
   return tweet


#loading data
data_sentiment = pd.read_csv('id_tweet_sentiment.csv')
#cleaning the tweets
tweet_clean = [clean_tweet(tweet) for tweet in data_sentiment.tweet_text]
data_sentiment['cleaned_tweet'] = tweet_clean
data_sentiment = data_sentiment[['Tweet_id','cleaned_tweet', 'sentiment']]


data_sentiment.to_csv('id_tweet_sentiment_cleaned.csv')

