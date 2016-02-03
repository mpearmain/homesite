# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc
import datetime

# settings
projPath = '/Users/konrad/Documents/projects/homesite/'
dataset_version = "ensemble_base"
todate = datetime.datetime.now().strftime("%Y%m%d")    
no_bags = 10
    
## data
# read the training and test sets
#xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')
xtrain = pd.read_csv(projPath + 'input/xvalid_20160202.csv')
id_train = xtrain.QuoteNumber
ytrain = xtrain.QuoteConversion_Flag
xtrain.drop('QuoteNumber', axis = 1, inplace = True)
xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)

#xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')
xtest = pd.read_csv(projPath + 'input/xfull_20160202.csv')
id_test = xtest.QuoteNumber
xtest.drop('QuoteNumber', axis = 1, inplace = True)

# Get rid of incorrect names for xgboost (scv-rbf) cannot handle '-'
xtrain = xtrain.rename(columns=lambda x: x.replace('-', ''))
xtest = xtest.rename(columns=lambda x: x.replace('-', ''))

sample = pd.read_csv(projPath + 'input/sample_submission.csv')

pred_average = True

for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=697,
                            nthread=-1,
                            max_depth=8,
                            min_child_weight = 8.4966878568341748,
                            learning_rate= 0.017668218154585136,
                            silent=True,
                            subsample=0.85005533637305219,
                            colsample_bytree=0.80496791490053055,
                            gamma=0.00040708117094670357,
                            seed=k*100+22)
     
                                        
    clf.fit(xtrain, ytrain, eval_metric="auc")
    preds = clf.predict_proba(xtest)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags


sample.QuoteConversion_Flag = pred_average
todate = datetime.datetime.now().strftime("%Y%m%d")    
dataset_version = '20160202'
sample.to_csv(projPath + 'submissions/xgb_meta_data'+dataset_version+'_'+str(no_bags)+'bag_'+todate+'.csv', index=False)