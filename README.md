# Real or Not? NLP with Disaster Tweets

Big-Scale Analytics 2020

Professor: Michalis Vlachos  <br /> 
Teaching Assistant: Ahmad Ajalloeian

## Overview ##


## Models ##

Embedding Layer from Keras and Dense Layer <br />

GloVe Embeddings and LSTM <br />
Glove Embeddings and Bidirectional LSTM <br />
Glove Embeddings and CNN LSTM <br />

Universal Sentence Encoder and Dense Layers <br />
Universal Sentence Encoder and CNN LSTM <br />
Universal Sentence Encoder and SVM <br />

Bidirectional Encoder Representations from Transformers <br /> 

## Dataset ##
**Files** <br />
train.csv - the training set <br />
test.csv - the test set <br />

**Columns** <br />
id - a unique identifier for each tweet <br />
text - the text of the tweet <br />
location - the location the tweet was sent from (may be blank) <br />
keyword - a particular keyword from the tweet (may be blank) <br />
target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0) <br />

## Current results ##
We achieved the best result with Universal Sentence Encoder and SVM: 0.82719. 
