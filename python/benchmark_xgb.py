# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import numpy as np
import xgboost as xgb

print('Loading Train data set')
x_train = pd.read_csv('input/xtrain_full.csv')
print('Loading Test data set')
test = pd.read_csv('input/xtest.csv')

y_train = x_train.QuoteConversion_Flag.values
x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

x_train = x_train.fillna(-1)
test = test.fillna(-1)

pred_average = True
no_bags = 3
for k in range(no_bags):
    clf = xgb.XGBClassifier(n_estimators=1791,
                            nthread=-1,
                            max_depth=46,
                            gamma=0.13899995466416837,
                            min_child_weight=7,
                            learning_rate=0.045377489290770984,
                            silent=False,
                            max_delta_step=0.087327665468280302,
                            subsample=0.88826206707102551,
                            colsample_bytree=0.71864366655117162,
                            seed = k*100+22)
    xgb_model = clf.fit(x_train, y_train, eval_metric="auc")
    preds = clf.predict_proba(test)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags

sample = pd.read_csv('input/sample_submission.csv')
sample.QuoteConversion_Flag = pred_average
sample.to_csv('output/xgb_homesite_3bag_11112015.csv', index=False)