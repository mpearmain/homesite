# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:20:32 2015

@author: konrad
"""

import numpy as np
import pandas as pd
from itertools import product
import datetime
import xgboost as xgb


if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "mp1"
    model_type = "xgb" 
    seed_value = 123
    todate = datetime.datetime.now().strftime("%Y%m%d")
    	    
    ## data
    # read the training and test sets
    xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')     
    id_train = xtrain.QuoteNumber
    ytrain = xtrain.QuoteConversion_Flag
    xtrain.drop('QuoteNumber', axis = 1, inplace = True)
    xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)
    
    xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')     
    id_test = xtest.QuoteNumber
    xtest.drop('QuoteNumber', axis = 1, inplace = True)
        
    # folds
    xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
    # work with 5-fold split
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))
    
    ## model
    # parameter grids: LR + range of training subjects to subset to 
    child_weight = [1, 4]
    max_depth = [6, 15]
    colsample = [0.73, 0.83]
    rowsample = [0.76, 0.81]
    gamma_val = [0, 0.001]
    eta_val = [0.03, 0.01]
    ntrees = [1000, 2000]
    param_grid = tuple([child_weight, max_depth, colsample, rowsample, gamma_val, eta_val, ntrees])
    param_grid = list(product(*param_grid))

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", param_grid[i]
            # configure model with j-th combo of parameters
            x = param_grid[i]
            clf = xgb.XGBClassifier(n_estimators=x[6],
                                    nthread=-1,
                                    max_depth=x[1],
                                    min_child_weight=x[0],
                                    learning_rate=x[5],
                                    silent=True,
                                    subsample=x[3],
                                    colsample_bytree=x[2],
                                    gamma=x[2],
                                    seed=seed_value)
            
            # loop over folds - Keeping as pandas for ease of use with xgb wrapper
            for j in range(1, n_folds+1):
                print "Processing Fold", j
                idx0 = xfolds[xfolds.fold5 != j].index
                idx1 = xfolds[xfolds.fold5 == j].index
                x0 = xtrain[xtrain.index.isin(idx0)]
                x1 = xtrain[xtrain.index.isin(idx1)]
                y0 = ytrain[ytrain.index.isin(idx0)]
                y1 = ytrain[ytrain.index.isin(idx1)]

                # fit the model on observations associated with subject whichSubject in this fold
                clf.fit(x0, y0, eval_metric="auc")
                mvalid[idx1,i] = clf.predict_proba(x1)[:,1]

            # fit on complete dataset
            clf.fit(xtrain, ytrain, eval_metric="auc")
            mfull[:,i] = clf.predict_proba(xtest)[:,1]
            
        
    ## store the results
    # add indices etc
    mvalid = pd.DataFrame(mvalid)
    mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
    mvalid['QuoteNumber'] = id_train
    mvalid['QuoteConversion_Flag'] = ytrain
    
    mfull = pd.DataFrame(mfull)
    mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
    mfull['QuoteNumber'] = id_test
    

    # save the files            
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    