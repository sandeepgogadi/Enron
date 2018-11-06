import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit

# Features List
features_list = ["poi"]

# Load the dataset into dictionary
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

# Dataset Structure

# No of employees
print "No of employees:", "\n", len(data_dict.keys()), "\n"

# List of employees
print "Employees List:", "\n", data_dict.keys(), "\n"

# Employee Information
print "Information Fields:", "\n", data_dict[data_dict.keys()[0]], "\n"

# Some of the information fields are not known and are filled as "NaN"

# Check for outliers

# Plotting salary vs bonus

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
print "Check out the plot:", "\n", "Close plot to continue", "\n"
plt.show()

# Outliers can be clearly seen

# To find them
for key in data_dict.keys():
    if float(data_dict[key]["salary"]) > 25000000 and float(data_dict[key]["bonus"]) > 80000000:
        print "Outlier:", "\n", key, "\n"

# TOTAL appears which is a document upload error
# To remove it
data_dict.pop('TOTAL', 0)

# Reformat the data
data = featureFormat(data_dict, features)

# Remove "NaN" that we earlier mentioned from data
outliers = []
for key in data_dict:
    val = data_dict[key]["salary"]
    if val == "NaN":
        continue
    else:
        outliers.append([int(val), key])

# Replot salary vs bonus to check for further outliers
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
print "Check out the plot:", "\n", "Close plot to continue", "\n"
plt.show()


# Check top 4 salaried Employees
outliers_list = sorted(outliers, reverse=True)[:4]
print "Top 4 salaried employees", "\n", outliers_list, "\n"

# We donot want to remove them because some of them are poi's
for i in range(4):
    key = outliers_list[i][1]
    print key, "is poi:", data_dict[key]["poi"]

print "\n"
# Have to look at more features or create new
# function to remove "NaN" and replace them with 0 for emails
def dict_to_list(key,normalizer):
    return_list = []
    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            return_list.append(0.)
        elif data_dict[i][key]>=0:
            return_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return return_list

# Create new features
# Two new features are fraction_to_poi_email, fraction_from_poi_email

# Create two lists of new features
fraction_from_poi_email = dict_to_list("from_poi_to_this_person", "to_messages")
fraction_to_poi_email = dict_to_list("from_this_person_to_poi", "from_messages")

# Add these two features into data_dict
count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1

# Add new features to features_list
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"]

# Store dataset into a new variable
my_dataset = data_dict

# Extract features from dataset
data = featureFormat(my_dataset, features_list)

# Plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.ylabel("fraction of emails this person sends to poi")
print "Check out the plot:", "\n", "Close plot to continue", "\n"
plt.show()

# Further we need to find efficient features
# To find them we apply Decision tree to rank them by score
# Then we can use the score and our intuition to remove ineffective features

# Add all features to features_list
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

# Reformat data
data = featureFormat(my_dataset, features_list)

# Split the data into features and labels
labels, features = targetFeatureSplit(data)

# Split data into training and testing datasets
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

# Apply Decision Tree Classifier
from time import time
t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
score = clf.score(features_test, labels_test)
print "accuracy:", "\n", score, "\n"
print "Decision Tree Time:", round(time()-t0,3), "s", "\n"

# Rank features based on their importance score
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print "length:", len(features_list)
print "Feature Ranking:"
for i in range(16):
    print "{} feature {} ({})".format(i+1, features_list[i+1], importances[indices[i]])
print "\n"

'''
10 features that I picked are
features_list = ["salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email", 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value']
The accuracy for this feature set is 0.8
'''

# Try out other algorithms
# Naive Bayes
from time import time
t0 = time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "accuracy:", accuracy, "\n"
print "Naive Bayes time: ", round(time()-t0,3), "s", "\n"

# Naive Bayes accuracy is comparitively much lower than Decision tree
# So select Decision Tree and improve its accuracy by further tuning its parameters
from time import time
t0 = time()

for min_samples_split in [2, 3, 4, 5, 6, 7]:
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    score = clf.score(features_test, labels_test)
    print "min_samples_split:", min_samples_split
    print "accuracy:", score

print "Decision Tree Time:", round(time()-t0,3), "s", "\n"

# min_samples_split = 5 gave 0.8
# So clf = DecisionTreeClassifier(min_samples_split=5)

# Analysis, Validation & Performance

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "shared_receipt_with_poi"]
my_dataset = data_dict
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

# Use KFold to split and validate algorithm
kf = cross_validation.KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

# Decision tree
t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'Accuracy before tuning:', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"


### use manual tuning parameter min_samples_split
t0 = time()
clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))

acc=accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "accuracy after tuning = ", acc

# function for calculation ratio of true positives
# out of all positives (true + false)
print 'precision = ', precision_score(labels_test,pred)

# function for calculation ratio of true positives
# out of true positives and false negatives
print 'recall = ', recall_score(labels_test,pred)


### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
