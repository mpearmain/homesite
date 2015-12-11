# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:20:32 2015

@author: konrad
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from itertools import product
import datetime
import sys
sys.path.append("/Users/konrad/Documents/projects/xgboost/wrapper/")
import xgboost as xgb


if __name__ == '__main__':

    ## settings
    projPath = '/Users/konrad/Documents/projects/homesite/' 
    dataset_version = "kb4"
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
    child_weight = [1,2,4]         
    max_depth = [10, 20]
    colsample = [0.6, 0.8, 0.9]
    rowsample = [0.6, 0.8, 0.9]
    gamma_val = [0, 0.001, 0.01]
    eta_val = [0.05, 0.01, 0.005, 0.0025]
    ntrees = [500, 1000]
    param_grid = tuple([child_weight, max_depth, colsample, rowsample,gamma_val, eta_val, ntrees ])
    param_grid = list(product(*param_grid))

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", i
            # configure model with j-th combo of parameters
            x = param_grid[i]
            xgb_param = {"objective":"multi:softprob","eval_metric":"mlogloss", 
            "num_class":3, "min_child_weight":x[0], 
              "max_depth":x[1], "colsample_bytree":x[2],
             "subsample":x[3] , "gamma":x[4], "eta":x[5], "silent":1}            
            
            # loop over folds
            for j in range(0,n_folds):
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0]; x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(ytrain)[idx0]; y1 = np.array(ytrain)[idx1]
			
                # fit the model on observations associated with subject whichSubject in this fold
                bst1 = xgb.train(params = xgb_param, dtrain = xgb.DMatrix(x0, label=y0), 
                                 num_boost_round = x[6]) 
                mvalid[idx1,i] = bst1.predict(xgb.DMatrix(x1))
                
            # fit on complete dataset
            bst1 = xgb.train(params = xgb_param, dtrain = xgb.DMatrix(xtrain, label=ytrain), 
                                 num_boost_round = x[6]) 
            mfull[:,i] = bst1.predict(xgb.DMatrix(xtest))
            
        
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
    