#!/usr/bin/env python
# coding: utf-8

##################################################################
#       PRACTICE PROJECT via data-flair.training
#
#       COMMENT-NOTATION
#       ###=> TASK
#         #=> TEST CODE FOR PRACTICE, NOT RELEVANT TO FINAL CODE
##################################################################

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

###read data
df=pd.read_csv('news.csv')
#df.shape
###Print text of first row : df.text[0]

###get labels
labels=df.label
#labels.head()
###Split the data
x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=7)


### Initialize a TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

### Fir and transform train set and also transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


### Initialize PassiveAggressiveClassifier
pc=PassiveAggressiveClassifier(max_iter=50)
pc.fit(tfidf_train,y_train)

### Predict on test set and calculate efficiency
y_pred=pc.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: { round(score*100,2)}%') #used f-string here, nice

### Confusion matrix build and print

conmat=confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
print(conmat)
##################################################################
# OUTPUT FORMAT
# ACCURACY_PERCENTAGE%
# ARRAY([[TRUE_POSITIVES,    TRUE_NEGATIVES]
#        [FALSE_POSITIVES,   FALSE_NEGATIVES]], DTYPE=x)
##################################################################
