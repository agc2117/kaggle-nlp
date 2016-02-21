import os
import sys
import pandas as pd
import part_1_code

sys.path.append(os.path.join('..'))

train = pd.read_csv('../data/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)

test = pd.read_csv('../data/testData.tsv', header = 0, delimiter = '\t', quoting = 3)

meaningful_train_data = []

for x in train['review']:
	meaningful_train_data.append(part_1_code.review_to_words(x))

meaningful_test_data = []

for x in test['review']:
	meaningful_test_data.append(part_1_code.review_to_words(x))

train_features, train_vectorizer = part_1_code.feature_extraction(meaningful_train_data)
test_features, test_vectorizer = part_1_code.feature_extraction(meaningful_test_data)

forest = part_1_code.train_random_forest(train_features, train['sentiment'])

result = forest.predict(test_features)

output = pd.DataFrame(data = {'id': test['id'], 'sentiment': result})
output.to_csv('../data/Bag_of_Words_Model.csv', index = False, quoting = 3)
