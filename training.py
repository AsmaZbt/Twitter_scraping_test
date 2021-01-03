# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from numba import jit, cuda 
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from timeit import default_timer
from sklearn import preprocessing
import nest_asyncio
nest_asyncio.apply()

RANDOM_SEED = 42

# count the embeddings with function 1 or 2 
TRANSFORMER_BATCH=128
def count_embeddings_1 (df):
    idx_chunk=list(df.columns).index('cleaned_tweet')
    embeddings_list = []
    for index in range (0, df.shape[0], TRANSFORMER_BATCH):
        embedds = model.encode(df.iloc[index:index+TRANSFORMER_BATCH, idx_chunk].values, show_progress_bar=False)
        embeddings_list.append(embedds)
    return np.concatenate(embeddings_list)

@jit
#(target ="cuda") 
def count_embeddings_2(df):
    df_list =df['cleaned_tweet']
    sentences = df_list.tolist()
    embeddings_list = model.encode(sentences)
    return embeddings_list



if __name__ == '__main__':


    #loading the Sentence Bert transformer
    model = SentenceTransformer("distiluse-base-multilingual-cased") # other existing Models (Ar): XLM-RoBERTa mean pooling / LASER/ mBERT mean pooling

    #split data into training/testing set
    df_sentiment = pd.read_csv('id_tweet_sentiment_cleaned.csv')
    df_train, df_test = train_test_split(df_sentiment,test_size=0.2, random_state= RANDOM_SEED)
    df_val, df_test = train_test_split(df_test,test_size=0.5, random_state= RANDOM_SEED)

    # sentence embeddings for training dataset
    start_time = default_timer()
    training_embeddings = count_embeddings_2(df_train)
    print("Train embeddings: {}: in: {:5.2f}s".format(training_embeddings.shape, default_timer() - start_time))

    # sentence embeddings for testing dataset
    start_time = default_timer()
    tested_embeddings = count_embeddings_2(df_test)
    print("Test embeddings: {}: in: {:5.2f}s".format(tested_embeddings.shape, default_timer() - start_time))

    X_train = np.array(train_embedd)
    X_test = np.array(tested_embeddings)
    X_train.shape, X_test.shape

    #Encoding Sentiment features

    enc = preprocessing.OneHotEncoder()
    label = df_train['sentiment'].values.reshape ((-1,1))
    enc.fit(label)
    y_train = enc.transform(label).toarray()
    #y_train = enc.inverse_transform(y_train)
    

    # need to check the Configuration of the Network classifier  !
    KERAS_VALIDATION_SPLIT=0.05
    KERAS_EPOCHS=70  
    KERAS_BATCH_SIZE=128 

    # Create and train Keras model
    n_features=X_train.shape[1]
    n_labels = y_train.shape[1]

    start_time=default_timer()

    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(2048, input_dim=n_features, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(n_labels, activation='softmax')
    ])

    LR=0.0001
    adam = keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=KERAS_EPOCHS, batch_size=KERAS_BATCH_SIZE, validation_split=KERAS_VALIDATION_SPLIT)

    print("Training the classifer. Dataset size: {} {:5.2f}s".format(X_train.shape, default_timer() - start_time))

    output_preds = model.predict(X_test)

    samples = df_test['sentiment']
    samples['Sentiment_pred'] = np.argmax(output_preds,axis=1)
    df_test['predicted_sent'] = samples['Sentiment_pred']

    df_test.to_csv("/content/drive/MyDrive/Colab Notebooks/sentiment_corpus/test_prediction.csv", index=False)



