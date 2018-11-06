#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    for i in range(len(ages)):
        cleaned_data.append((int(ages[i]), float(net_worths[i]), abs(float(net_worths[i] - predictions[i]))))

    cleaned_data = sorted(cleaned_data, key=lambda c : c[2])

    return cleaned_data[:80]
