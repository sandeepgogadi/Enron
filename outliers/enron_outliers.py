#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

data_salary, data_bonus = targetFeatureSplit(data)

bandits = []
for i in range(len(data_salary)):
        if float(data_salary[i]) > 1000000 and float(data_bonus[i]) > 5000000:
            bandits.append([float(data_salary[i]), float(data_bonus[i])])

print bandits


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
