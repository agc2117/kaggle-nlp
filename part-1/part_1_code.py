import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
import re

sys.path.append(os.path.join('..'))

def review_to_words(text):
	import nltk
	from nltk.corpus import stopwords
	#
	#Remove HTML
	review_text = BeautifulSoup(text).get_text()
	#
	#Remove non-letters
	letters_only = re.sub("[^a-zA-Z]", " ", review_text)
	#
	#Convert to lower and split
	words = letters_only.lower().split()
	#
	#Convert stopwords to set for speed
	stops = set(stopwords.words('english'))
	#
	#Clean stopwords from text
	meaningful_words = [w for w in words if w not in stops]
	#
	#Join back into string
	return " ".join(meaningful_words)

def feature_extraction(text):
	#
	#Use Scikit-Learn to convert the text to feature vectors
	from sklearn.feature_extraction.text import CountVectorizer
	#
	#Set max num features; preprocessor and tokenizer can be changed
	vectorizer = CountVectorizer(analyzer = 'word', preprocessor = None, tokenizer = None, stop_words = None, max_features = 5000)
	#
	#Create feature vectors and convert to array
	features = vectorizer.fit_transform(text).toarray()
	#
	#Return the feature list and the model to the user
	return features, vectorizer

def train_random_forest(examples, labels):
	from sklearn.ensemble import RandomForestClassifier
	#
	#Use 100 trees as default for speed & efficiency
	forest = RandomForestClassifier(n_estimators = 100)

	forest = forest.fit(examples, labels)

	return forest


