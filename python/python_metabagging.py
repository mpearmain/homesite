import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
import random
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier

random.seed(21)
np.random.seed(21)

LINES = 61877

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)


    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.log(1+X))
    rbm1 = SVC(C=100.0, gamma = 0.1, probability=True, verbose=1).fit(X[0:9999,:], y[0:9999])
    rbm2 = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto', bootstrap=False, oob_score=False, n_jobs=-1, verbose=1).fit(X[0:9999,:], y[0:9999])
    rbm3 = GradientBoostingClassifier(n_estimators=50,max_depth=11,subsample=0.8,min_samples_leaf=5,verbose=1).fit(X[0:9999,:], y[0:9999])
    X =  np.append(X[10000:LINES,:], np.power(rbm1.predict_proba(X[10000:LINES,:])*rbm2.predict_proba(X[10000:LINES,:])*rbm3.predict_proba(X[10000:LINES,:]), (1/3.0))   , 1)
    return X, y[10000:LINES], encoder, scaler, rbm1, rbm2, rbm3

def load_test_data(path, scaler, rbm1, rbm2, rbm3):
    df = pd.read_csv(path)
    X = df.values.copy()

    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(np.log(1+X))
    X =  np.append(X, np.power(rbm1.predict_proba(X)*rbm2.predict_proba(X)*rbm3.predict_proba(X), (1/3.0)), 1)
    return X, ids

def make_submission(y_prob, ids, encoder, name='/home/mikeskim/Desktop/kaggle/otto/data/lasagneSeed21.csv'):
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))


#Load Data
X, y, encoder, scaler, rbm1, rbm2, rbm3 = load_train_data('/home/mikeskim/Desktop/kaggle/otto/data/train.csv')
X_test, ids = load_test_data('/home/mikeskim/Desktop/kaggle/otto/data/test.csv', scaler, rbm1, rbm2, rbm3)

num_classes = len(encoder.classes_)
num_features = X.shape[1]

print(num_classes); print(num_features); print(X)


layers0 = [('input', InputLayer),
('dropoutf', DropoutLayer),
('dense0', DenseLayer),
('dropout', DropoutLayer),
('dense1', DenseLayer),
('dropout2', DropoutLayer),
('dense2', DenseLayer),
('output', DenseLayer)]


net0 = NeuralNet(layers=layers0,

input_shape=(None, num_features),
dropoutf_p=0.1,
dense0_num_units=600,
dropout_p=0.3,
dense1_num_units=600,
dropout2_p=0.1,
dense2_num_units=600,

output_num_units=num_classes,
output_nonlinearity=softmax,

update=adagrad,
update_learning_rate=0.008,
eval_size=0.2,
verbose=1,
max_epochs=20)



net0.fit(X, y)
y_prob = net0.predict_proba(X_test)
num_runs = 50

for jj in xrange(num_runs):
  print(jj)
  X, y, encoder, scaler, rbm1, rbm2, rbm3 = load_train_data('/home/mikeskim/Desktop/kaggle/otto/data/train.csv')
  X_test, ids = load_test_data('/home/mikeskim/Desktop/kaggle/otto/data/test.csv', scaler, rbm1, rbm2, rbm3)
  num_classes = len(encoder.classes_)
  num_features = X.shape[1]
  net0.fit(X, y)
  y_prob = y_prob + net0.predict_proba(X_test)


y_prob = y_prob/(num_runs+1.0)
make_submission(y_prob, ids, encoder)