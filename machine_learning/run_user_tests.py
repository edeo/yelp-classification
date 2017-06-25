##########################
#This scipt uses AWS to test different users and the efficacy of the recommendation system 
#########################

#Import neceessary modules

import json
import pandas as pd
import re
import random
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import sys
sys.path.append('../machine_learning')
import yelp_ml as yml
reload(yml)
from gensim import corpora, models, similarities, matutils
import tqdm

#Load in word dictionaries and the large user JSON file
lh_neg = open('../input/negative-words.txt', 'r').read()
lh_neg = lh_neg.split('\n')
lh_pos = open('../input/positive-words.txt', 'r').read()
lh_pos = lh_pos.split('\n')
users = json.load(open("cleaned_large_user_dictionary.json"))
word_list = list(set(lh_pos + lh_neg))

#####Test Machine Learning Algorithms
conn = MongoClient()
db = conn.get_database('cleaned_data')
reviews = db.get_collection('restaurant_reviews')
users_results = {}

for j in tqdm.tqdm(range(0, len(users.keys()))):
    user_df = yml.make_user_df(users[users.keys()[j]])
    business_ids = list(set(user_df['biz_id']))
    restreview = {}

    for i in range(0, len(business_ids)):
        rlist = []
        for obj in reviews.find({'business_id':business_ids[i]}):
            rlist.append(obj)
        restreview[business_ids[i]] = rlist

    restaurant_df = yml.make_biz_df(users.keys()[j], restreview)
    
    #Create a training and test sample from the user reviewed restaurants
    split_samp = .25
    len_random = int(len(business_ids) * split_samp)
    test_set = random.sample(business_ids, len_random)
    training_set = [x for x in business_ids if x not in test_set]
    sub_train_reviews, train_labels, train_reviews, train_ratings = [], [], [], []

    #Create a list of training reviews and training ratings
    for rest_id in training_set:
        train_reviews.append((user_df[user_df['biz_id'] == rest_id]['review_text'].iloc[0],
                                 user_df[user_df['biz_id'] == rest_id]['rating'].iloc[0]))

    #Create an even sample s.t. len(positive_reviews) = len(negative_reviews)
    sample_size = min(len([x[1] for x in train_reviews if x[1] < 4]),
                          len([x[1] for x in train_reviews if x[1] >= 4]))
    
    bad_reviews = [x for x in train_reviews if x[1] < 4]
    good_reviews = [x for x in train_reviews if x[1] >= 4]

    for L in range(0, int(float(sample_size)/float(2))):
        sub_train_reviews.append(bad_reviews[L][0])
        sub_train_reviews.append(good_reviews[L][0])
        train_labels.append(bad_reviews[L][1])
        train_labels.append(good_reviews[L][1])
        
    #Make the train labels binary
    train_labels = [1 if x >=4 else 0 for x in train_labels]

    #Make a FeatureUnion object with the desired features then fit to train reviews
    test_results = {}
    feature_selection = {"sent_tf":(True, True, False), 
                         "sent": (True,False,False),
                         "tf_lda": (False,True,True), 
                         "all": (True, True, True)}

    for feature in feature_selection.keys():
        #Make a FeatureUnion object with the desired features then fit to train reviews
        comb_features = yml.make_featureunion(sent_percent=feature_selection[feature][0], 
                                              tf = feature_selection[feature][1], 
                                              lda = feature_selection[feature][2])
        
        delta_vect = None
        comb_features.fit(sub_train_reviews)
        train_features = comb_features.transform(sub_train_reviews)

        #Fit LSI model and return number of LSI topics
        lsi, topics, dictionary = yml.fit_lsi(sub_train_reviews)
        train_lsi = yml.get_lsi_features(sub_train_reviews, lsi, topics, dictionary)

        #Stack the LSI and combined features together
        train_features = sparse.hstack((train_features, train_lsi))
        train_features = train_features.todense()

        #fit each model in turn 
        model_runs = {"svm": (True, False, False),
                      "rf": (False, True, False), 
                      "naive_bayes": (False, False, True)}

        for model_run in model_runs.keys():
            clf = yml.fit_model(train_features, train_labels, svm_clf = model_runs[model_run][0], 
                            RandomForest = model_runs[model_run][1], 
                                nb = model_runs[model_run][2])
            threshold = 0.7
            error = yml.test_user_set(test_set, clf, restaurant_df, user_df, comb_features, 
                                      threshold, lsi, topics, dictionary, delta_vect)
            test_results[(feature, model_run)] = (yml.get_log_loss(error), 
                                            yml.get_accuracy_score(error), 
                                            yml.get_precision_score(error))
    
    string_keys_dict = {}
    for key in test_results.keys():
        string_keys_dict[str(key)] = test_results[key]
            
with open('test_results.json', 'wb') as fp:
    json.dump(string_keys_dict, fp)