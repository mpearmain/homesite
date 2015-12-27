# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:44:27 2015

@author: konrad
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from itertools import product
import datetime

if __name__ == '__main__':

    ## settings
    projPath = '/Users/konrad/Documents/projects/homesite/' 
    dataset_version = "kb3"
    model_type = "logreg" 
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
    
    #xtrain.drop('SalesField8', axis = 1, inplace = True)
    #xtest.drop('SalesField8', axis = 1, inplace = True)
    
    # folds
    xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
    # work with 5-fold split
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))
    
    ## model
    # setup model instances
    model = LogisticRegression()
           
    # parameter grids
    c_vals = [0.01, 0.05, 0.1, 0.25, 0.9]         
    pen_vals = ['l1']                                
    f_vals = [ True]                                 
    c_weights = ['auto']
    param_grid = tuple([c_vals,pen_vals,f_vals,c_weights])
    param_grid = list(product(*param_grid))

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", i
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model.loss = x[0]; model.penalty = x[1]; model.l1_ratio = x[2]; 
            model.alpha = x[3]            
            
            # loop over folds
            for j in range(0,n_folds):
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0]; x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(ytrain)[idx0]; y1 = np.array(ytrain)[idx1]
			
                # fit the model on observations associated with subject whichSubject in this fold
                model.fit(x0, y0)
                mvalid[idx1,i] = model.predict_proba(x1)[:,1]
                print "finished fold:", j
                
            # fit on complete dataset
            model.fit(xtrain, ytrain)
            mfull[:,i] = model.predict_proba(xtest)[:,1]
            print "finished full prediction"
        
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
    