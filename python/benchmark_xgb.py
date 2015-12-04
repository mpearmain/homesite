# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb

print('Loading Train data set')
x_train = pd.read_csv('input/xtrain_full.csv')
print('Loading Test data set')
test = pd.read_csv('input/xtest.csv')

sample = pd.read_csv('input/sample_submission.csv')

y_train = x_train.QuoteConversion_Flag.values
x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

x_train = x_train.fillna(-1)
test = test.fillna(-1)

pred_average = True
no_bags = 10
for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=900,
                            nthread=-1,
                            max_depth=8,
                            learning_rate=0.03,
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=0.85,
                            seed=k*100+22)
    xgb_model = clf.fit(x_train, y_train, eval_metric="auc")
    preds = clf.predict_proba(test)[:,1]


    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags

sample.QuoteConversion_Flag = pred_average
sample.to_csv('output/xgb_homesite_10bag_19112015.csv', index=False)
