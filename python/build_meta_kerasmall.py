"""
Created on Thu Dec 10 10:44:27 2015

@author: konrad
"""
from keras.regularizers import l2, activity_l2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from itertools import product
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "kb4"
    model_type = "keras" 
    seed_value = 123
    todate = datetime.datetime.now().strftime("%Y%m%d")
    np.random.seed(seed_value) 

    # data
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
    nb_classes = 2
    dims = xtrain.shape[1]
    epochs = 15
    
    # parameter grids
    drop_vals = [0.1, 0.01]
    dec_vals = [0.8, 0.99]
    lr_vals = [0.1, 0.5, 0.25]
    reg_vals = [1e-5, 1e-3]
    lay_vals = [1]
    param_grid = tuple([drop_vals, dec_vals, lr_vals, reg_vals, lay_vals])
    param_grid = list(product(*param_grid))
    
    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", param_grid[i]
            print "Combo:", i, "of", len(param_grid)
            # loop over folds
            # Recompile model on each fold
            for j in range(0,n_folds):
                # configure model with j-th combo of parameters
                x = param_grid[i]
                model = Sequential()
                model.add(Dense(dims * x[4], input_shape=(dims,),W_regularizer=l2(x[3])))
                #model.add(PReLU())
                model.add(BatchNormalization())
                model.add(Dropout(x[0]))
                model.add(Dense(nb_classes))
                model.add(Activation('softmax'))
                opt=Adadelta(lr=x[2],decay=x[1],epsilon=1e-5)
                model.compile(loss='binary_crossentropy', optimizer=opt)

                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0]
                x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(ytrain)[idx0]
                y1 = np.array(ytrain)[idx1]
                y00 = np.zeros((x0.shape[0],2))
                y00[:,0] = y0; y00[:,1] = 1-  y0
                # fit the model on observations associated with subject whichSubject in this fold
                model.fit(x0, y00, nb_epoch=epochs, batch_size=1000)
                mvalid[idx1,i] = model.predict_proba(x1)[:,0]
                del model
                print "finished fold:", j

            print "Building full prediction model for test set."
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model = Sequential()
            model.add(Dense(dims * x[4], input_shape=(dims,),W_regularizer=l2(x[3])))
            #model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(x[0]))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))
            opt=Adadelta(lr=x[2],decay=x[1],epsilon=1e-5)
            model.compile(loss='binary_crossentropy', optimizer=opt)

            # fit on complete dataset
            ytrain0 = np.zeros((xtrain.shape[0],2))
            ytrain0[:,0] = ytrain
            ytrain0[:,1] = 1- ytrain
            model.fit(np.array(xtrain), ytrain0,nb_epoch=epochs, batch_size=1000)
            mfull[:,i] = model.predict_proba(np.array(xtest))[:,0]
            del model
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
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + 'mini_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + 'mini_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    