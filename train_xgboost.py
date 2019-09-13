import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
import sys
filename = sys.argv[1] # "307_data.csv"
dataset = pd.read_csv(filename)

# y SalePrice
y = dataset.pop("y")

X = pd.get_dummies(dataset)

trainx, testx ,trainy, testy = train_test_split(X,y,test_size=0.2,random_state=33)

dtrain = xgb.DMatrix(trainx, label=trainy)
dtest = xgb.DMatrix(testx, label=testy)

params = {
            # Parameters that we are going to tune.
            'max_depth':6,
            'min_child_weight': 1,
            'eta':.3,
            'subsample': 1,
            'colsample_bytree': 1,
                                    # Other parameters
            'objective':'reg:linear',
         }

bst = xgb.XGBRegressor(params)
print("Training:", bst)
bst.fit(trainx,trainy)
print("Testing:")
preds = bst.predict(testx)

from sklearn.metrics import r2_score
print(testy, preds)
print(r2_score(testy, preds))

# 0.92 / 1.0
