# coding: utf-8
__author__ = 'mpearmain'


# models gives 0.9816

import pandas as pd
import xgboost as xgb

print('Loading Train data set')
x_train = pd.read_csv('input/xtrain_kb5.csv')
print('Loading Test data set')
test = pd.read_csv('input/xtest_kb5.csv')

sample = pd.read_csv('input/sample_submission.csv')

y_train = x_train.QuoteConversion_Flag.values
x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

x_train = x_train.fillna(-1)
test = test.fillna(-1)

pred_average = True
no_bags = 3
for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=400,
                            nthread=-1,
                            max_depth=39,
                            learning_rate=0.1054,
                            silent=True,
                            subsample=0.78667,
                            colsample_bytree=0.67303,
                            max_delta_step=0.33433188,
                            seed=k*100+22)
                            #seed=1234)
    xgb_model = clf.fit(x_train, y_train, eval_metric="auc")
    preds = clf.predict_proba(test)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags

print('Writing Submission file...')
sample.QuoteConversion_Flag = pred_average
sample.to_csv('output/xgb_homesite_3bag_kb5_29112015.csv', index=False)