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
from sklearn.ensemble import RandomForestClassifier
from sklearn_deltatfidf import DeltaTfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn_deltatfidf import DeltaTfidfVectorizer
from sklearn import svm
import math
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities, matutils
#################################
#Declare import word dictionaries
#################################

lh_neg = open('../input/negative-words.txt', 'r').read()
lh_neg = lh_neg.split('\n')
lh_pos = open('../input/positive-words.txt', 'r').read()
lh_pos = lh_pos.split('\n')

pos_vectorizer = CountVectorizer(vocabulary = lh_pos)
neg_vectorizer = CountVectorizer(vocabulary = lh_neg)
stop_words = set(stopwords.words('english'))

#################################
#Feature objects and functions
#################################

def sent_percent(review):
    regex_words = re.compile('[a-z]+')
    words = [x.lower() for x in review.split(' ')]
    words = [x for x in words if regex_words.match(x)]
    pos_count, neg_count = 0, 0
    for word in words:
        if word in lh_pos:
            pos_count += 1
        elif word in lh_neg:
            neg_count += 1
    return [float(pos_count)/float(len(words)), float(neg_count)/float(len(words))]


class SentimentPercentage(BaseEstimator, TransformerMixin):
    """Takes in two lists of strings, extracts the lev distance between each string, returns list"""

    def __init__(self):
        pass

    def transform(self, reviews):
        ##Take in a list of textual reviews and return a list with two elements:
        ##[Positive Percentage, Negative Percentage]
        pos_vect = pos_vectorizer.transform(reviews)
        neg_vect = neg_vectorizer.transform(reviews)
        features = []
        
        for i in range(0, len(reviews)):
            sent_percentage = []
            sent_percentage.append(float(pos_vect[i].sum())/float(len(reviews[i])))
            sent_percentage.append(float(neg_vect[i].sum())/float(len(reviews[i])))
            features.append(sent_percentage)
            
        return np.array(features)

    def fit(self, reviews, y=None, n_grams = None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class TfIdfGramTransformer(BaseEstimator, TransformerMixin):
    """Takes in two lists of strings, extracts the lev distance between each string, returns list"""

    def __init__(self):
        pass

    def transform(self, reviews):
        tf_vector = vectorizer.transform(reviews)
        return tf_vector

    def fit(self, reviews, y=None, n_grams = (0,2)):
        vectorizer = TfidfVectorizer(ngram_range = n_grams, stop_words = 'english')
        vectorizer.fit(reviews)
        """Returns `self` unless something different happens in train and test"""
        return vectorizer

def get_restaurant_reviews(ip, business_ids):
    #Input: IP is the ip to the AWS instance that has MongoDB running
    #business_ids is a list of unique businesses IDs for which the user has created reviews
    #Output: A dictionary where the restaurant IDs are the keys and the entries to each key is the list of reviews for that business

    conn = MongoClient(ip, 27017)
    conn.database_names()
    db = conn.get_database('cleaned_data')
    reviews = db.get_collection('restaurant_reviews')

    restreview = {}

    for i in range(0, len(business_ids)):
        rlist = []
        for obj in reviews.find({'business_id':business_ids[i]}):
            rlist.append(obj)
        restreview[business_ids[i]] = rlist

    return restreview

def fit_lsi(train_reviews):
    #Input: train_reviews is a list of reviews that will be used to train the LSI feature transformer
    #Output: A trained LSI model and the transformed training reviews

    texts = [[word for word in review.lower().split() if (word not in stop_words)]
              for review in train_reviews]
    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    numpy_matrix = matutils.corpus2dense(corpus, num_terms=10000)
    singular_values = np.linalg.svd(numpy_matrix, full_matrices=False, compute_uv=False)
    mean_sv = sum(list(singular_values))/len(singular_values)
    topics = int(mean_sv)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topics)

    return lsi, topics, dictionary

def make_featureunion(sent_percent=True, tf = True, lda = True):
    #Input: sent_percent, tf, and lda are all boolean variables and indicate which features should be used in the ML algorithm
    #sent_percent: The percentage of positive and negative words in a review, as defiend by H&L (2004), relative to the total words
    #tf: tf-idf representation using a ngram range of (0,1)
    #lda: LDA representation using an ngram range of (1,1)
    #Output: A FeatureUnion object with the specified features horizontally stacked in a SciPy Sparse Matrix.

    if sent_percent == False:
        comb_features = FeatureUnion([('tf', TfIdfGramTransformer()),
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (1,1))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=50))]))
                             ])
    elif tf == False:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()), 
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (1,1))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=50))]))
                             ])
    elif lda == False:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()),('tf', TfIdfGramTransformer())
                             ])
    else:
        comb_features = FeatureUnion([('sent_percent',SentimentPercentage()),('tf', TfIdfGramTransformer()), 
                              ('lda', Pipeline([('bow', TfidfVectorizer(stop_words='english', ngram_range = (1,1))), 
                                        ('lda_transform', LatentDirichletAllocation(n_topics=50))])),
                              ('tf_bow', TfidfVectorizer(stop_words='english'))
                             ])

    return comb_features


def fit_model(train_features, train_labels, svm_clf = False, RandomForest = False, nb = False):
    #Input: SVM, RandomForest, and NB are all boolean variables and indicate which model should be fitted
    #SVM: Linear Support Vector Machine
    #RandomForest: Random Forest, we set the max_depth equal to 50 because of prior tests
    #NB: LDA representation using an ngram range of (1,1)
    #train_features: Train reviews that have been transformed into the relevant features
    #train_labels: Labels for the training reviews, transformed into a binary variable
    #Output: A fitted model object

    if svm_clf == True:
        clf = svm.LinearSVC()
        clf.fit(train_features, train_labels)
        return clf
    elif RandomForest == True:
        clf = RandomForestClassifier(max_depth = 100, max_leaf_nodes=50, criterion='entropy')
        clf.fit(train_features, train_labels)
        return clf
    elif nb == True:
        clf = GaussianNB()
        clf.fit(train_features, train_labels)
        return clf
    else:
        return None

def fit_features(train_reviews, comb_features):
    #Input:
    #train_reviews: A list of reviews to be fitted upon 
    #comb_features: A FeatureUnion object with the relevant features specified
    #Output: A fitted FeatureUnion object

    if not train_reviews:
        return None
    else:
        comb_features = comb_features.fit(train_reviews)

    return comb_features

def get_lsi_features(reviews, lsi, topics, dictionary):
    #Input:
    #reviews: A list of reviews to be transformed
    #lsi: A fitted LSI model 
    #topics: An integer, number of topics that the LSI model was fitted upon
    #Output: A matrix of features that have been transformed according to the LSI model

    if not reviews:
        return None
    else:
        texts = [[word for word in review.lower().split() if (word not in stop_words)]
              for review in reviews]
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        reviews_tfidf = tfidf[corpus]
        reviews_lsi = lsi[reviews_tfidf]
        reviews_lsi = [[text[1] for text in review] for review in reviews_lsi]
        reviews_lsi = [[0.000000000001] * topics if len(x) != topics else x for x in reviews_lsi]
        reviews_lsi = sparse.coo_matrix(reviews_lsi)

    return reviews_lsi

def get_transformed_features(comb_features, reviews):
    #Input: 
    #comb_features: A FeatureUnion object which will be used to create a stacked feature matrix
    #reviews: A list of reviews to be transformed
    #Output: A feature matrix of the reviews, transformed according to the features in the FeatureUnion object

    if not reviews:
        return None
    else:
        transformed_reviews = comb_features.transform(reviews)

    return transformed_reviews

def make_biz_df(user_id, restreview):
    #Input: 
    #user_id: A specific customer ID
    #restreview: A dictionary where the keys are the restaurant IDs and entries 
    #are a list of the reviews for that restaurant
    #Output: A dataframe with the columns (review_text, rest_ratings, biz_ids)
    rest_reviews = []
    rest_ratings = []
    biz_ids = []
    for i in range(0, len(restreview.keys())):
        for restaurant in restreview[restreview.keys()[i]]:
            if restaurant['user_id'] != user_id:
                rest_reviews.append(restaurant['text'])
                rest_ratings.append(restaurant['stars'])
                biz_ids.append(restreview.keys()[i])
            else:
                pass
    rest_reviews = [review.encode('utf-8').translate(None, string.punctuation) for review in rest_reviews]

    biz_df = pd.DataFrame({'review_text': rest_reviews, 'rating': rest_ratings, 'biz_id': biz_ids})

    return biz_df

def make_user_df(user_specific_reviews):
    #Input: 
    #user_specific_reviews: A list of reviews for a specific user
    #Output: A dataframe with the columns (user_reviews, user_ratings, biz_ids)
    user_reviews = []
    user_ratings = []
    business_ids = []

    for review in user_specific_reviews:
        user_reviews.append(review['text'])
        user_ratings.append(review['stars'])
        business_ids.append(review['business_id'])

    ###WE SHOULD MAKE THE OUR OWN PUNCTUATION RULES
    #https://www.tutorialspoint.com/python/string_translate.htm
    user_reviews = [review.encode('utf-8').translate(None, string.punctuation) for review in user_reviews]
        
    user_df = pd.DataFrame({'review_text': user_reviews, 'rating': user_ratings, 'biz_id': business_ids})
    return user_df

def test_user_set(test_set, clf, restaurant_df, users_df, comb_features, threshold, lsi = None, topics = None, dictionary = None, delta_tfidf = None):
    #Input: 
    #test_set: The set of restaurant IDs, split from the users total set, on which we will test our classifier
    #clf: Classifier trained on the fully stacked features 
    #restaurant_df: A dataframe of the restaurants in the test_set and the reviews associated with each restaurant
    #users_df: A dataframe of the users reviews with the tuple (user, restaurant id, restaurant rating, review)
    #comb_features: A FeatureUnion object that has been trained on the user's other reviews
    #threshold: A float value 
    #Note: lsi and topics should be GLOBAL variables after running fit_lsi
    #Output: A list of errors on predicting whether or not the user likes the restaurant in the test set
    comb_error = []
    for i in range(0,len(test_set)):
        predicted_rating = 0
        #Get reviews for that restaurant
        test_reviews =[]
        
        test_reviews.extend(list(restaurant_df[restaurant_df['biz_id'] == test_set[i]]['review_text']))
        #Transform features
        test_features = comb_features.transform(test_reviews)
        #LSI Features
        

        #Stack the features
        if lsi == None or topics == None or dictionary == None:
            stacked_test_features = test_features.todense()
        elif delta_tfidf != None:
            test_lsi = get_lsi_features(test_reviews, lsi, topics, dictionary)
            test_delta_tfidf = delta_tfidf.transform(test_reviews)
            stacked_test_features = sparse.hstack((test_features, test_lsi, test_delta_tfidf))
            stacked_test_features =  stacked_test_features.todense()
        else:
            test_lsi = get_lsi_features(test_reviews, lsi, topics, dictionary)
            stacked_test_features = sparse.hstack((test_features, test_lsi))
            stacked_test_features =  stacked_test_features.todense()

        #Get ML prediction
        test_prediction = clf.predict(stacked_test_features)

        if test_prediction.mean() >= threshold:
            predicted_rating = 1

        actual_rating = list(users_df[users_df['biz_id'] == test_set[i]]['rating'])[0]

        if actual_rating >= 4:
            actual_rating = 1
        else:
            actual_rating = 0

        comb_error.append((test_prediction, predicted_rating, actual_rating))
    return comb_error

def get_top_ten_recs(test_predictions):
    #Input: 
    #test_predictions: A list of tuples in the form (Confidence, Classification Prediction) for the user's set of test restaurants
    #Output: A list of the top 10 restaurants that the classification algorithm is most confident in
    if test_predictions:
        confidence_tuple = [(float(sum(list(x[0])))/float(len(x[0])),x[1]) for x in test_predictions]
        confidence_tuple.sort()
        top_ten = confidence_tuple[-10:]
        return top_ten
    else:
        print "Empty List Passed"
        return None

def get_log_loss(test_predictions):
    #Input: 
    #test_predictions: A list of tuples in the form (Confidence, Classification Prediction) for the user's set of test restaurants
    #Output: The log loss score associated with each classifier
    if test_predictions:
        test_predictions = [(np.append(x[0],0.0000000000001),x[1], x[2]) if x[0].mean() == 0.0 else x for x in test_predictions]
        test_predictions = [(np.append(x[0],0.0000000000001),x[1], x[2]) if x[0].mean() == 1 else x for x in test_predictions]
        raw_log = [x[1] * math.log(x[0].mean()) + (1-x[2]) * math.log(1-x[0].mean()) for x in test_predictions]
        log_loss = float(sum(raw_log))/float(len(raw_log))
        return log_loss
    else:
        print "Empty List Passed"
        return None

def get_precision_score(test_predictions):
    #Input: 
    #test_predictions: A list of tuples in the form (Confidence, Classification Prediction) for the user's set of test restaurants
    #Output: The accuracy score associated with each classifier
    if test_predictions:
        true_positive = [x for x in test_predictions if (x[1] == x[2]) & (x[1] == 1)]
        false_positive = [x for x in test_predictions if (x[1] != x[2]) & (x[1] == 1)]
        try:
            precision_score = float(len(true_positive))/float(len(true_positive) + len(false_positive))
        except:
            precision_score = 0.0
        return precision_score
    else:
        print "Empty List Passed"
        return None

def get_accuracy_score(test_predictions):
    #Input: 
    #test_predictions: A list of tuples in the form (Confidence, Classification Prediction) for the user's set of test restaurants
    #Output: The accuracy score associated with each classifier
    if test_predictions:
        test_errors = [abs(x[1]-x[2]) for x in test_predictions]
        accuracy_score = float(sum(test_errors))/float(len(test_errors))
        return accuracy_score
    else:
        print "Empty List Passed"
        return None
