from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
from math import floor
import joblib
#creating data set from csv
coloumn_names = ['data','target_name']

dataset = pd.read_csv("train.csv", names = coloumn_names, nrows = 100)

print(dataset)
#classes of categories
dataset_new = dataset.replace({
    'target_name' : {
        'comment':0,
        'service':1,
        'aviation':2,
        'request':3,
        'maintainance':4,
        'airline_authority':6,
        'security':7
    }
})

dataset['target'] = dataset_new['target_name']
classes = list(set(dataset['target_name']))
print(dataset)

#splitting dataset
def train_test_split(dataset, train_split):
    dataset_len = len(dataset)
    train_split = train_split/100.0
    split_percentage = floor(dataset_len*train_split)
    return dataset.data[0:split_percentage], dataset.data[split_percentage:], dataset.target[0:split_percentage], dataset.target[split_percentage:]

train_split_percentage = 70
x_train, x_test, y_train, y_test = train_test_split(dataset,train_split_percentage)
# print(x_train)
# print()
# print(y_train)


count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)

# print(f'(Documents, words) => {x_train_counts.shape}')
#
# print(x_train_counts.toarray())

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# print(x_train_tfidf.toarray())
model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

#finding the class
doc_new = ['@BLRAirport Today i visited to Bangalore air india took too much time ']
x_new_counts = count_vect.transform(doc_new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)
for doc, category in zip(doc_new , predicted):
    print(f'{doc} => {classes[category]}')

x_test_cv = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_cv)

result = model.predict(x_test_tfidf)

accuracy = np.mean(result == np.array(y_test))
print(f'accuracy: {accuracy*100}%')

filename = 'categorizer.pkl'
joblib.dump(model,filename)
