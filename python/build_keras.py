import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics

'''
    Keras Deep Learning model for ROC AUC score (local 5-fold validation)

'''


def float32(k):
    return np.cast['float32'](k)

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, init='uniform'))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta")

    return model


if __name__ == "__main__":
    # Load data set and target values

    DATASETS_TRAIN = ['input/xtrain_kb4.csv']
    DATASETS_TEST = ['input/xtest_kb4.csv']
    SEEDS = [1234]

    print('Loading X-folds data set')
    x_folds = pd.read_csv('input/xfolds.csv')

    # Read submission file for later to construct testPredict.
    submission = pd.read_csv('input/sample_submission.csv')

    # Create the loop structure.
    for i in xrange(len(DATASETS_TRAIN)):
        for seed in SEEDS:
            # Read the data files in - Train, test, and xfolds.
            print 'Loading', DATASETS_TRAIN[i], 'data set'
            x_train = pd.read_csv(DATASETS_TRAIN[i])
            print 'Loading', DATASETS_TEST[i], 'data set'
            test = pd.read_csv(DATASETS_TEST[i])
            test = test.drop('QuoteNumber', axis=1)

            # Make sure final set has no na's (should be solved in data_preparation.R)
            x_train = x_train.fillna(-1)

            print 'Splitting data sets for train, and validiation'
            meta_quoteNum = x_folds[x_folds['valid'] == 0 ].QuoteNumber
            meta_train = x_train[x_train['QuoteNumber'].isin(meta_quoteNum)]
            meta_y = meta_train.QuoteConversion_Flag.values
            meta_train = meta_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
            meta_names = list(meta_train)
            del meta_quoteNum

            valid_quoteNum = x_folds[x_folds['valid'] == 1 ].QuoteNumber
            valid_train = x_train[x_train['QuoteNumber'].isin(valid_quoteNum)]
            valid_y = valid_train.QuoteConversion_Flag.values
            valid_train = valid_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
            valid_quoteNum
            print 'Done...'

            # Convert class vector to binary class matrix, for use with categorical_crossentropy
            Y = np_utils.to_categorical(meta_y)
            v_y = np_utils.to_categorical(valid_y)
            meta_train = meta_train.astype(np.float32)
            valid_train = valid_train.astype(np.float32)

            input_dim = meta_train.shape[1]
            output_dim = 2

            print("Validation...")

            model = build_model(input_dim, output_dim)

            print("Training model...")

            model.fit(meta_train, Y, nb_epoch=10, batch_size=32)
            valid_preds = model.predict_proba(valid_train, verbose=0)
            valid_preds = valid_preds[:, 1]
            roc = metrics.roc_auc_score(valid_train, valid_preds)
            print("ROC:", roc)

