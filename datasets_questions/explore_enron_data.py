#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data.keys())
print len(enron_data.values()[145])

print "POI"
count = 0
for i in range(146):
    if enron_data.values()[i]["poi"] == True:
        count += 1
print count

print "names"
f = open("../final_project/poi_names.txt", 'r')
names = f.readlines()
names = names[2:]
print len(names)
f.close()

print (enron_data['PRENTICE JAMES']['total_stock_value'])
print (enron_data['COLWELL WESLEY'])

print "Colwell Wesley"
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print "Total Payments"
print enron_data['LAY KENNETH L']['total_payments']
print enron_data['SKILLING JEFFREY K']['total_payments']
print enron_data['FASTOW ANDREW S']['total_payments']

print "Salary", "Email"
nsalary = 0
nemail = 0
npayments = 0
poi = False
npoi = 0
npoipayments = 0
for i in enron_data.keys():
    salary = enron_data[i]['salary']
    email = enron_data[i]['email_address']
    payments = enron_data[i]['total_payments']
    poi = enron_data[i]['poi']
    nsalary += 0 if salary == 'NaN' else 1
    nemail += 0 if email == 'NaN' else 1
    npayments += 1 if payments == 'NaN' else 0
    npoi += 1 if poi == True else 0
    npoipayments += 1 if (payments == 'NaN' and poi == True) else 0
print nsalary, nemail, npayments, npoi, npoipayments

people = len(enron_data.keys())

print 100*npayments/people
