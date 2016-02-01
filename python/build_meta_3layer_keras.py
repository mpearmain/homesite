import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.layers.core import Dense
import datetime

## settings
projPath = './'
dataset_version = "mp4"
model_type = "keras"
seed_value = 578943
todate = datetime.datetime.now().strftime("%Y%m%d")
np.random.seed(seed_value)
np.random.seed(1778)  # for reproducibility
need_normalise=True
need_categorical=False

def getDummy(df,col):
    category_values=df[col].unique()
    data=[[0 for i in range(len(category_values))] for i in range(len(df))]
    dic_category=dict()
    for i,val in enumerate(list(category_values)):
        dic_category[str(val)]=i
   # print dic_category
    for i in range(len(df)):
        data[i][dic_category[str(df[col][i])]]=1

    data=np.array(data)
    for i,val in enumerate(list(category_values)):
        df.loc[:,"_".join([col,str(val)])]=data[:,i]

    return df

train = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')
id_train = train.QuoteNumber
y_train_quote_flag = train.QuoteConversion_Flag
y_train = train.QuoteConversion_Flag
train.drop('QuoteNumber', axis = 1, inplace = True)
train.drop('QuoteConversion_Flag', axis = 1, inplace = True)
train.drop(['PropertyField6', 'GeographicField10A'], axis=1, inplace = True)

test = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')
id_test = test.QuoteNumber
test.drop('QuoteNumber', axis = 1, inplace = True)
test.drop(['PropertyField6', 'GeographicField10A'], axis=1, inplace = True)

for f in test.columns:#
    if train[f].dtype=='object':
        lbl = LabelEncoder()
        lbl.fit(list(train[f])+list(test[f]))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

#try to encode all params less than 100 to be category
if need_categorical:
    #row bind train and test
    x = train.append(test, ignore_index=True)
    del train
    del test
    for f in x.columns:#
        category_values= set(list(x[f].unique()))
        if len(category_values) < 4:
            print (f)
            x = getDummy(x, f)
    test = x.iloc[260753:, ]
    train = x.iloc[:260753:, ]

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train).astype(np.int32)
y_train = np_utils.to_categorical(y_train)

print ("processsing finished")
train = np.array(train)
train = train.astype(np.float32)
test = np.array(test)
test = test.astype(np.float32)
if need_normalise:
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

# folds
xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
# work with 5-fold split
fold_index = xfolds.fold5
fold_index = np.array(fold_index) - 1
n_folds = len(np.unique(fold_index))

nb_classes = 2
print nb_classes, 'classes'

dims = train.shape[1]
print dims, 'dims'

auc_scores=[]
best_score=-1

param_grid = [[1024, 0.1, 0.6, 1024, 0.6, 420, 0.6, 400],
              [1324, 0.15, 0.6, 712, 0.8, 520, 0.7, 400],
              [96, 0.05, 0.4, 1512, 0.4, 330, 0.6, 400]]

# storage structure for forecasts
mvalid = np.zeros((train.shape[0],len(param_grid)))
mfull = np.zeros((test.shape[0],len(param_grid)))

## build 2nd level forecasts
for i in range(len(param_grid)):
        print "processing parameter combo:", param_grid[i]
        print "Combo:", i+1, "of", len(param_grid)
        # loop over folds
        # Recompile model on each fold
        for j in range(0,n_folds):
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model = Sequential()
            model.add(Dense(x[0], input_shape=(dims,)))
            model.add(Dropout(x[1]))# input dropout
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(x[2]))
            model.add(Dense(x[3]))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(x[4]))
            model.add(Dense(x[5]))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(x[6]))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))
            model.compile(loss='binary_crossentropy', optimizer="sgd")

            idx0 = np.where(fold_index != j)
            idx1 = np.where(fold_index == j)
            x0 = np.array(train)[idx0,:][0]
            x1 = np.array(train)[idx1,:][0]
            y0 = np.array(y_train)[idx0]
            y1 = np.array(y_train)[idx1]

            # fit the model on observations associated with subject whichSubject in this fold
            model.fit(x0, y0, nb_epoch=x[7], batch_size=1256)
            mvalid[idx1,i] = model.predict_proba(x1)[:,1]
            y_pre = model.predict_proba(x1)[:,1]
            scores = roc_auc_score(y1[:,1],y_pre)
            print 'AUC score', scores
            del model
            print "finished fold:", j

        print "Building full prediction model for test set."
        # configure model with j-th combo of parameters
        x = param_grid[i]
        model = Sequential()
        model.add(Dense(x[0], input_shape=(dims,)))
        model.add(Dropout(x[1]))#    input dropout
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(x[2]))
        model.add(Dense(x[3]))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(x[4]))
        model.add(Dense(x[5]))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(x[6]))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer="sgd")
        # fit on complete dataset

        model.fit(np.array(train), y_train, nb_epoch=x[7], batch_size=1256)
        mfull[:,i] = model.predict_proba(np.array(test))[:,1]

        del model
        print "finished full prediction"

## store the results
# add indices etc
mvalid = pd.DataFrame(mvalid)
mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
mvalid['QuoteNumber'] = id_train
mvalid['QuoteConversion_Flag'] = y_train_quote_flag

mfull = pd.DataFrame(mfull)
mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
mfull['QuoteNumber'] = id_test


# save the files
mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
