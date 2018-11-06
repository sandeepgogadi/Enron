#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here

### it's all yours from here forward!

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 0)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
pred = clf.predict(features_test)

for i in range(len(pred)):
    if pred[i] != labels_test[i]:
        print pred[i], labels_test[i]

from sklearn.metrics import precision_score, recall_score

precision = precision_score(labels_test, pred, average=None)
recall = recall_score(labels_test, pred, average=None)

print precision, recall
