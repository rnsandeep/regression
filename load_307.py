import csv
import sys
import numpy as np
import pandas as pd 

def z_score(val):
    mean = np.mean(val)
    std = np.std(val)
    return (val-mean)/std

def gaussian_outlier(val):
    mean, std = np.mean(val), np.std(val)
#    print(mean)
    low_cut, high_cut = mean-std*3, mean+std*3
#    print(low_cut, high_cut, mean)
    outliers_lower = np.where(val < low_cut)[0]
    outliers_high = np.where(val > high_cut)[0]
    outliers = list(outliers_lower)+ list(outliers_high)
    return outliers
#def countNAN(val):
#    for v in val:


data = pd.read_csv(sys.argv[1]) 




#head = data.head(10)
cols = list(data.columns.values)
#print(len(data.index))
out = []
nans = []
print(data.shape)
for col in cols:
    val = data[col]
    print(col, np.mean(val))

    nan_count  = np.where(np.isnan(val))[0].shape[0]
#    print(nan_count, col)
    nans.append(nan_count)

#    outliers = gaussian_outlier(val)
#    out =  out+ outliers
#    out = list(set(outliers)&set(out))

from collections import Counter
#count  =  Counter(out)
#for i in range(0, 30000, 1000):
#   print("no of columns where nans are not more than 10000", np.where(np.array(nans) < i)[0].shape[0], i)
#print(count)
#print(len(out))
#print(dir(data))
#print(data.column_name)

#print(head)

