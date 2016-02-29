# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:08:00 2015

@author: konrad
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from itertools import product
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "QC300"
    model_type = "etrees" 
    seed_value = 789
    todate = datetime.datetime.now().strftime("%Y%m%d")
    	    
    ## data
    # read the training and test sets HACK
    xtrain = pd.read_csv(projPath + 'input/train.csv')
    id_train = xtrain.QuoteNumber
    y_train_quote_flag = xtrain.QuoteConversion_Flag
    y_train = xtrain.QuoteConversion_Flag
    xtrain.drop('QuoteNumber', axis = 1, inplace = True)
    xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)
    xtrain.drop(['PropertyField6', 'GeographicField10A'], axis=1, inplace = True)

    xtest = pd.read_csv(projPath + 'input/test.csv')
    id_test = xtest.QuoteNumber
    xtest.drop('QuoteNumber', axis = 1, inplace = True)
    xtest.drop(['PropertyField6', 'GeographicField10A'], axis=1, inplace = True)

    xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')
    xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)
    xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')
    xtest.drop('QuoteConversion_Flag', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
    # work with 5-fold split
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))
    
    ## model
    # setup model instances
    model = ExtraTreesClassifier(criterion='gini', 
                                 max_depth=None, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_leaf_nodes=None, 
                                 bootstrap=False, oob_score=False,
                                 n_jobs= -1,
                                 random_state= seed_value, 
                                 verbose=1, warm_start=False, 
                                 class_weight=None)

      
    # parameter grids: LR + range of training subjects to subset to 
    n_vals = [250]
    n_minleaf = [1]
    n_minsplit = [1]
    n_maxfeat = [0.1]
    param_grid = tuple([n_vals, n_minleaf, n_minsplit, n_maxfeat])
    param_grid = list(product(*param_grid))

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", i
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model.n_estimators = x[0]
            model.min_samples_leaf = x[1]     
            model.min_samples_split = x[2]
            model.max_features = x[3]
            
            # loop over folds
            for j in range(0,n_folds):
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0];
                x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(y_train)[idx0];
                y1 = np.array(y_train)[idx1]

                # fit the model on observations associated with subject whichSubject in this fold
                model.fit(x0, y0)
                mvalid[idx1,i] = model.predict_proba(x1)[:,1]
                y_pre = model.predict_proba(x1)[:,1]
                scores = roc_auc_score(y1,y_pre)
                print 'AUC score', scores
                print "finished fold:", j
                
            # fit on complete dataset
            model.fit(xtrain, y_train)
            mfull[:,i] = model.predict_proba(xtest)[:,1]
            print "finished full prediction"
            
    ## store the results
    # add indices etc
    mvalid = pd.DataFrame(mvalid)
    mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
    mvalid['QuoteNumber'] = id_train
    mvalid['QuoteConversion_Flag'] = y_train
    
    mfull = pd.DataFrame(mfull)
    mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
    mfull['QuoteNumber'] = id_test
    

    # save the files            
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    