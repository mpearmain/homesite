# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:45:37 2015

@author: konrad
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score as auc
from itertools import product
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "kb5"
    model_type = "svc-rbf"
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
    # setup model instances
    model = BaggingClassifier(base_estimator=SVC(probability = True, random_state= seed_value),
                              n_estimators=8,
                              n_jobs=8,
                              max_samples = 20000,
                              max_features = 0.9)
           
    # parameter grids: LR + range of training subjects to subset to 
    c_vals = [0.1, 1, 10]
    g_vals = [1, 2, 5]
    c_weights = ['auto']
    param_grid = tuple([c_vals, c_weights, g_vals])
    param_grid = list(product(*param_grid))

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", param_grid[i]
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model.C = x[0]
            model.class_weight = x[1]
            model.gamma = x[2]
            
            # loop over folds
            for j in range(0,n_folds):
                print "Running fold", j+1
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0];
                x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(ytrain)[idx0];
                y1 = np.array(ytrain)[idx1]
			
                # fit the model on observations associated with subject whichSubject in this fold
                model.fit(x0, y0)
                mvalid[idx1,i] = model.predict_proba(x1)[:,1]
                
            # fit on complete dataset
            model.fit(xtrain, ytrain)
            mfull[:,i] = model.predict_proba(xtest)[:,1]
            
        
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
    