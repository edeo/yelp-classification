import json
import pandas as pd
import re
from scipy import sparse
import numpy as np
from pymongo import MongoClient
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from gensim import corpora, models, similarities, matutils
import tqdm

def make_featureunion(sent_percent=True, tf = True, lda = True):
    if sent_percent == False:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()),('tf', TfIdfGramTransformer()), 
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (0,2))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=500))]))
                             ])
    elif tf == False:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()),('tf', TfIdfGramTransformer()), 
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (0,2))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=500))]))
                             ])
    elif lda == False:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()),('tf', TfIdfGramTransformer()), 
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (0,2))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=500))]))
                             ])
    elif lsi == False:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()),('tf', TfIdfGramTransformer()), 
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (0,2))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=500))]))
                             ])
    return comb_features



comb_features.fit(train_reviews)
train_features = comb_features.transform(train_reviews)
train_features = sparse.hstack((train_features, train_lsi))

#XGBoost training
#gbm = xgb.XGBClassifier(max_depth=10, n_estimators=500, learning_rate=0.02, ).fit(train_features, train_labels)
#RandomForest training
#rf = RandomForestClassifier()
#rf.fit(train_features, train_labels)
#SVM training
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(train_features, train_labels)
comb_error = []
for i in tqdm.tqdm(range(0,len(test_set))):
    predicted_rating = 0
    #Get reviews for that restaurant
    test_reviews =[]
    test_reviews.extend(list(restaurant_df[restaurant_df['biz_id'] == test_set[i]]['review_text']))
    
    #Transform features
    test_features = comb_features.transform(test_reviews)
    
    #LSI Features
    test_texts = [[word for word in review.lower().split() if (word not in stop_words)]
          for review in test_reviews]
    test_corpus = [dictionary.doc2bow(test) for test in test_texts]
    test_tfidf = tfidf[test_corpus]
    test_lsi = lsi[test_tfidf]
    test_lsi = [[test[1] for test in test_review] for test_review in test_lsi]
    test_lsi = [[0.0000000001] * topics if len(x) != topics else x for x in test_lsi]
    
    test_lsi = sparse.coo_matrix(test_lsi)
    stacked_test_features = sparse.hstack((test_features, test_lsi))

    #Get XGBoost prediction
    #test_prediction = gbm.predict(stacked_test_features)
    #Get SVM prediction
    test_prediction = svm_classifier.predict(stacked_test_features)
    #Get Random Forest prediction
    #test_prediction = rf.predict(stacked_test_features)   

    if test_prediction.mean() > 0.5:
        predicted_rating = 1

    actual_rating = list(user_df[user_df['biz_id'] == test_set[i]]['rating'])[0]
    if actual_rating >= 4:
        actual_rating = 1
    else:
        actual_rating = 0

    comb_error.append(abs(predicted_rating - actual_rating))
    