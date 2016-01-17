# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc

## data
# read the training and test sets
xtrain = pd.read_csv('input/xtrain_ensemble_base.csv')
id_train = xtrain.QuoteNumber
ytrain = xtrain.QuoteConversion_Flag
xtrain.drop('QuoteNumber', axis = 1, inplace = True)
xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)

xtest = pd.read_csv('input/xtest_ensemble_base.csv')
id_test = xtest.QuoteNumber
xtest.drop('QuoteNumber', axis = 1, inplace = True)

# Get rid of incorrect names for xgboost (scv-rbf) cannont handle '-'
xtrain = xtrain.rename(columns=lambda x: x.replace('-', ''))
xtest = xtest.rename(columns=lambda x: x.replace('-', ''))

sample = pd.read_csv('input/sample_submission.csv')

pred_average = True
no_bags = 20
for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=388,
                            nthread=-1,
                            max_depth=7,
                            learning_rate=0.0270118686,
                            silent=True,
                            subsample=0.8639708,
                            colsample_bytree=0.871944475,
                            gamma=0.000634406,
                            seed=k*100+22)
    clf.fit(xtrain, ytrain, eval_metric="auc")
    preds = clf.predict_proba(xtest)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags


sample.QuoteConversion_Flag = pred_average
sample.to_csv('output/xgb_meta_20bag_17012016.csv', index=False)