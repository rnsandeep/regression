import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
import sys
import numpy as np
from tqdm import tqdm 
filename = sys.argv[1] # "307_data.csv"
dataset = pd.read_csv(filename)

# y SalePrice
y = dataset.pop("y")

X = pd.get_dummies(dataset)

cols = list(dataset.columns.values)
#rows = list(dataset.rows.values)
def z_score(val):
    mean = np.mean(val)
    std = np.std(val)
    return (val-mean)/std

def mean_norm(val):
    mean= np.mean(val)
    return val-mean/(np.max(val)-np.min(val))



for col in cols:
    X[col] =  z_score(X[col]) #mean_norm(X[col]) #z_score(X[col])

removes = []
for row in tqdm(range(X.shape[0])):
    xrow = X.iloc[[row]]
    count  = np.where(np.isnan(xrow))[0].shape[0] #sum([1 for i in xrow if np.isnan(i)])
#    print(count)
    if count > 150:
      removes.append(row)
ind = X.index[[removes]]
X = X.drop(ind)
y = y.drop(ind)

print(X.shape)
print(X.head(10))    

df = pd.concat([X,y])


# 0.92 / 1.0
correlation = df.corr()

print(correlation)
