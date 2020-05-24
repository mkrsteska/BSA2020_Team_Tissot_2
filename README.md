# Real or Not? NLP with Disaster Tweets

Big-Scale Analytics 2020

Professor: Michalis Vlachos  <br /> 
Teaching Assistant: Ahmad Ajalloeian

## Overview ##
### Link to the video ### 
[Link](https://www.google.com)

## Structure of the project ##
###### In order to run the notebooks, please download the GloVe model from this link: https://drive.google.com/file/d/1jnyyMXLPUAY8Fh4cSXSeDzRnJHE-gKUz/view?usp=sharing and place it in the data folder. ######

###### In order to run the notebooks in Google Colab, please upload the file 'preprocess_tweets.py' in Google Colab. The file is placed in the code folder ######

#### code ####
BERT_2.ipynb <br />
Copie_de_BERT.ipynb <br />
GloVe Embeddings and LSTM.ipynb <br />
Project Notebook.ipynb <br />
Universal Sentence Encoder and Keras Sequential Model.ipynb <br />
Universal Sentence Encoder and SVM.ipynb <br />
preprocess_tweets.py

#### data ####
train.csv - the training set <br />
test.csv - the test set <br />
submission.csv - example of a submission file <br />
1. Submission_MultinomialNB.csv <br />
2. Submission_LSTM.csv <br />
3. Submission_Bidirectional_LSTM.csv <br />
4. Submission_CNN_LSTM.csv <br />
5. Submission_Tensorflow_Keras.csv <br />
6. Submission_Tensorflow_Keras.csv <br />
7. Submission_Tensorflow_Keras.csv <br />
8. Submission_SVC.csv <br />
9. Submission_Keras_Embeddings.csv <br />
10. Submission_Keras_Embeddings.csv <br />
11. Submission_BERT.csv <br />

## Models ##

###### Notebook 'Universal Sentence Encoder and LSTM.ipynb' ######
GloVe Embeddings and LSTM <br />
Glove Embeddings and Bidirectional LSTM <br />
Glove Embeddings and CNN LSTM <br />

###### Notebook 'Universal Sentence Encoder and Keras Sequential Model.ipynb' ######
Universal Sentence Encoder and Dense Layers <br />
Universal Sentence Encoder and CNN LSTM <br />

###### Notebook 'Universal Sentence Encoder and SVM.ipynb' ######
Universal Sentence Encoder and SVM <br />

###### Notebook 'BERT_2.ipynb' and 'Copie_de_BERT.ipynb' ######
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
