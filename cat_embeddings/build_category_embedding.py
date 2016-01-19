import pickle
import numpy
import math

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

from prepare_nn_features import split_features


class Model(object):

    def __init__(self, train_ratio):
        self.train_ratio = train_ratio
        self.__load_data()

    def evaluate(self):
        if self.train_ratio == 1:
            return 0
        total_sqe = 0
        num_real_test = 0
        for record, sales in zip(self.X_val, self.y_val):
            if sales == 0:
                continue
            guessed_sales = self.guess(record)
            sqe = ((sales - guessed_sales) / sales) ** 2
            total_sqe += sqe
            num_real_test += 1
        result = math.sqrt(total_sqe / num_real_test)
        return result

    def __load_data(self):
        f = open('feature_train_data.pickle', 'rb')
        (self.X, self.y) = pickle.load(f)
        self.X = numpy.array(self.X)
        self.y = numpy.array(self.y)
        self.num_records = len(self.X)
        self.train_size = int(self.train_ratio * self.num_records)
        self.test_size = self.num_records - self.train_size
        self.X, self.X_val = self.X[:self.train_size], self.X[self.train_size:]
        self.y, self.y_val = self.y[:self.train_size], self.y[self.train_size:]


class NN_with_EntityEmbedding(Model):

    def __init__(self, train_ratio):
        super().__init__(train_ratio)
        self.build_preprocessor(self.X)
        self.nb_epoch = 20
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = numpy.max(numpy.log(self.y))
        self.min_log_y = numpy.min(numpy.log(self.y))
        self.__build_keras_model()
        self.fit()

    def build_preprocessor(self, X):
        X_list = split_features(X)
        # Google trend de
        self.gt_de_enc = StandardScaler()
        self.gt_de_enc.fit(X_list[32])
        # Google trend state
        self.gt_state_enc = StandardScaler()
        self.gt_state_enc.fit(X_list[33])

    def preprocessing(self, X):
        X_list = split_features(X)
        X_list[32] = self.gt_de_enc.transform(X_list[32])
        X_list[33] = self.gt_state_enc.transform(X_list[33])
        return X_list

    def __build_keras_model(self):
        models = []

        model_Field6 = Sequential()
        model_Field6.add(Embedding(8, 50, input_length=1))
        model_Field6.add(Reshape(dims=(50,)))
        models.append(model_Field6)

        model_Field7 = Sequential()
        model_Field7.add(Embedding(28, 50, input_length=1))
        model_Field7.add(Reshape(dims=(50,)))
        models.append(model_Field7)

        model_Field8 = Sequential()
        model_Field8.add(Embedding(38, 50, input_length=1))
        model_Field8.add(Reshape(dims=(50,)))
        models.append(model_Field8)

        model_Field9 = Sequential()
        model_Field9.add(Embedding(5, 50, input_length=1))
        model_Field9.add(Reshape(dims=(50,)))
        models.append(model_Field9)

        model_Field10 = Sequential()
        model_Field10.add(Embedding(8, 50, input_length=1))
        model_Field10.add(Reshape(dims=(50,)))
        models.append(model_Field10)

        model_Field11 = Sequential()
        model_Field11.add(Embedding(11, 50, input_length=1))
        model_Field11.add(Reshape(dims=(50,)))
        models.append(model_Field11)

        model_Field12 = Sequential()
        model_Field12.add(Embedding(2, 50, input_length=1))
        model_Field12.add(Reshape(dims=(50,)))
        models.append(model_Field12)

        model_CoverageField1A = Sequential()
        model_CoverageField1A.add(Embedding(26, 50, input_length=1))
        model_CoverageField1A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField1A)

        model_CoverageField1B = Sequential()
        model_CoverageField1B.add(Embedding(26, 50, input_length=1))
        model_CoverageField1B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField1B)

        model_CoverageField2A = Sequential()
        model_CoverageField2A.add(Embedding(25, 50, input_length=1))
        model_CoverageField2A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField2A)

        model_CoverageField2B = Sequential()
        model_CoverageField2B.add(Embedding(25, 50, input_length=1))
        model_CoverageField2B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField2B)

        model_CoverageField3A = Sequential()
        model_CoverageField3A.add(Embedding(25, 50, input_length=1))
        model_CoverageField3A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField3A)

        model_CoverageField3B = Sequential()
        model_CoverageField3B.add(Embedding(25, 50, input_length=1))
        model_CoverageField3B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField3B)

        model_CoverageField4A = Sequential()
        model_CoverageField4A.add(Embedding(25, 50, input_length=1))
        model_CoverageField4A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField4A)

        model_CoverageField4B = Sequential()
        model_CoverageField4B.add(Embedding(25, 50, input_length=1))
        model_CoverageField4B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField4B)

        model_CoverageField5A = Sequential()
        model_CoverageField5A.add(Embedding(3, 50, input_length=1))
        model_CoverageField5A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField5A)

        model_CoverageField5B = Sequential()
        model_CoverageField5B.add(Embedding(4, 50, input_length=1))
        model_CoverageField5B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField5B)

        model_CoverageField6A = Sequential()
        model_CoverageField6A.add(Embedding(3, 50, input_length=1))
        model_CoverageField6A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField6A)

        model_CoverageField6B = Sequential()
        model_CoverageField6B.add(Embedding(4, 50, input_length=1))
        model_CoverageField6B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField6B)

        model_CoverageField8 = Sequential()
        model_CoverageField8.add(Embedding(7, 50, input_length=1))
        model_CoverageField8.add(Reshape(dims=(50,)))
        models.append(model_CoverageField8)

        model_CoverageField9 = Sequential()
        model_CoverageField9.add(Embedding(12, 50, input_length=1))
        model_CoverageField9.add(Reshape(dims=(50,)))
        models.append(model_CoverageField9)

        model_CoverageField11A = Sequential()
        model_CoverageField11A.add(Embedding(26, 50, input_length=1))
        model_CoverageField11A.add(Reshape(dims=(50,)))
        models.append(model_CoverageField11A)

        model_CoverageField11B = Sequential()
        model_CoverageField11B.add(Embedding(26, 50, input_length=1))
        model_CoverageField11B.add(Reshape(dims=(50,)))
        models.append(model_CoverageField11B)

        model_SalesField1A = Sequential()
        model_SalesField1A.add(Embedding(25, 50, input_length=1))
        model_SalesField1A.add(Reshape(dims=(50,)))
        models.append(model_SalesField1A)

        model_SalesField1B = Sequential()
        model_SalesField1B.add(Embedding(25, 50, input_length=1))
        model_SalesField1B.add(Reshape(dims=(50,)))
        models.append(model_SalesField1B)

        model_SalesField2A = Sequential()
        model_SalesField2A.add(Embedding(26, 50, input_length=1))
        model_SalesField2A.add(Reshape(dims=(50,)))
        models.append(model_SalesField2A)

        model_SalesField2B = Sequential()
        model_SalesField2B.add(Embedding(26, 50, input_length=1))
        model_SalesField2B.add(Reshape(dims=(50,)))
        models.append(model_SalesField2B)

        model_SalesField3 = Sequential()
        model_SalesField3.add(Embedding(2, 50, input_length=1))
        model_SalesField3.add(Reshape(dims=(50,)))
        models.append(model_SalesField3)

        model_SalesField4 = Sequential()
        model_SalesField4.add(Embedding(5, 50, input_length=1))
        model_SalesField4.add(Reshape(dims=(50,)))
        models.append(model_SalesField4)

        model_SalesField5 = Sequential()
        model_SalesField5.add(Embedding(5, 50, input_length=1))
        model_SalesField5.add(Reshape(dims=(50,)))
        models.append(model_SalesField5)

        model_SalesField6 = Sequential()
        model_SalesField6.add(Embedding(24, 50, input_length=1))
        model_SalesField6.add(Reshape(dims=(50,)))
        models.append(model_SalesField6)

        model_SalesField7 = Sequential()
        model_SalesField7.add(Embedding(7, 50, input_length=1))
        model_SalesField7.add(Reshape(dims=(50,)))
        models.append(model_SalesField7)

        model_SalesField8 = Sequential()
        model_SalesField8.add(Embedding(61530, 50, input_length=1))
        model_SalesField8.add(Reshape(dims=(50,)))
        models.append(model_SalesField8)

        model_SalesField9 = Sequential()
        model_SalesField9.add(Embedding(2, 50, input_length=1))
        model_SalesField9.add(Reshape(dims=(50,)))
        models.append(model_SalesField9)

        model_SalesField10 = Sequential()
        model_SalesField10.add(Embedding(19, 50, input_length=1))
        model_SalesField10.add(Reshape(dims=(50,)))
        models.append(model_SalesField10)

        model_SalesField11 = Sequential()
        model_SalesField11.add(Embedding(20, 50, input_length=1))
        model_SalesField11.add(Reshape(dims=(50,)))
        models.append(model_SalesField11)

        model_SalesField12 = Sequential()
        model_SalesField12.add(Embedding(22, 50, input_length=1))
        model_SalesField12.add(Reshape(dims=(50,)))
        models.append(model_SalesField12)

        model_SalesField13 = Sequential()
        model_SalesField13.add(Embedding(8, 50, input_length=1))
        model_SalesField13.add(Reshape(dims=(50,)))
        models.append(model_SalesField13)

        model_SalesField14 = Sequential()
        model_SalesField14.add(Embedding(12, 50, input_length=1))
        model_SalesField14.add(Reshape(dims=(50,)))
        models.append(model_SalesField14)

        model_SalesField15 = Sequential()
        model_SalesField15.add(Embedding(12, 50, input_length=1))
        model_SalesField15.add(Reshape(dims=(50,)))
        models.append(model_SalesField15)

        model_PersonalField1 = Sequential()
        model_PersonalField1.add(Embedding(2, 50, input_length=1))
        model_PersonalField1.add(Reshape(dims=(50,)))
        models.append(model_PersonalField1)

        model_PersonalField2 = Sequential()
        model_PersonalField2.add(Embedding(2, 50, input_length=1))
        model_PersonalField2.add(Reshape(dims=(50,)))
        models.append(model_PersonalField2)

        model_PersonalField4A = Sequential()
        model_PersonalField4A.add(Embedding(26, 50, input_length=1))
        model_PersonalField4A.add(Reshape(dims=(50,)))
        models.append(model_PersonalField4A)

        model_PersonalField4B = Sequential()
        model_PersonalField4B.add(Embedding(26, 50, input_length=1))
        model_PersonalField4B.add(Reshape(dims=(50,)))
        models.append(model_PersonalField4B)

        model_PersonalField5 = Sequential()
        model_PersonalField5.add(Embedding(9, 50, input_length=1))
        model_PersonalField5.add(Reshape(dims=(50,)))
        models.append(model_PersonalField5)

        model_PersonalField6 = Sequential()
        model_PersonalField6.add(Embedding(2, 50, input_length=1))
        model_PersonalField6.add(Reshape(dims=(50,)))
        models.append(model_PersonalField6)

        model_PersonalField7 = Sequential()
        model_PersonalField7.add(Embedding(3, 50, input_length=1))
        model_PersonalField7.add(Reshape(dims=(50,)))
        models.append(model_PersonalField7)

        model_PersonalField8 = Sequential()
        model_PersonalField8.add(Embedding(3, 50, input_length=1))
        model_PersonalField8.add(Reshape(dims=(50,)))
        models.append(model_PersonalField8)

        model_PersonalField9 = Sequential()
        model_PersonalField9.add(Embedding(3, 50, input_length=1))
        model_PersonalField9.add(Reshape(dims=(50,)))
        models.append(model_PersonalField9)

        model_PersonalField10A = Sequential()
        model_PersonalField10A.add(Embedding(26, 50, input_length=1))
        model_PersonalField10A.add(Reshape(dims=(50,)))
        models.append(model_PersonalField10A)

        model_PersonalField10B = Sequential()
        model_PersonalField10B.add(Embedding(26, 50, input_length=1))
        model_PersonalField10B.add(Reshape(dims=(50,)))
        models.append(model_PersonalField10B)

        model_PersonalField11 = Sequential()
        model_PersonalField11.add(Embedding(5, 50, input_length=1))
        model_PersonalField11.add(Reshape(dims=(50,)))
        models.append(model_PersonalField11)

        model_PersonalField12 = Sequential()
        model_PersonalField12.add(Embedding(5, 50, input_length=1))
        model_PersonalField12.add(Reshape(dims=(50,)))
        models.append(model_PersonalField12)

        model_PersonalField13 = Sequential()
        model_PersonalField13.add(Embedding(4, 50, input_length=1))
        model_PersonalField13.add(Reshape(dims=(50,)))
        models.append(model_PersonalField13)

        model_PersonalField14 = Sequential()
        model_PersonalField14.add(Embedding(30, 50, input_length=1))
        model_PersonalField14.add(Reshape(dims=(50,)))
        models.append(model_PersonalField14)

        model_PersonalField15 = Sequential()
        model_PersonalField15.add(Embedding(22, 50, input_length=1))
        model_PersonalField15.add(Reshape(dims=(50,)))
        models.append(model_PersonalField15)

        model_PersonalField16 = Sequential()
        model_PersonalField16.add(Embedding(50, 50, input_length=1))
        model_PersonalField16.add(Reshape(dims=(50,)))
        models.append(model_PersonalField16)

        model_PersonalField17 = Sequential()
        model_PersonalField17.add(Embedding(66, 50, input_length=1))
        model_PersonalField17.add(Reshape(dims=(50,)))
        models.append(model_PersonalField17)

        model_PersonalField18 = Sequential()
        model_PersonalField18.add(Embedding(61, 50, input_length=1))
        model_PersonalField18.add(Reshape(dims=(50,)))
        models.append(model_PersonalField18)

        model_PersonalField19 = Sequential()
        model_PersonalField19.add(Embedding(57, 50, input_length=1))
        model_PersonalField19.add(Reshape(dims=(50,)))
        models.append(model_PersonalField19)

        model_PersonalField22 = Sequential()
        model_PersonalField22.add(Embedding(7, 50, input_length=1))
        model_PersonalField22.add(Reshape(dims=(50,)))
        models.append(model_PersonalField22)

        model_PersonalField23 = Sequential()
        model_PersonalField23.add(Embedding(13, 50, input_length=1))
        model_PersonalField23.add(Reshape(dims=(50,)))
        models.append(model_PersonalField23)

        model_PersonalField24 = Sequential()
        model_PersonalField24.add(Embedding(14, 50, input_length=1))
        model_PersonalField24.add(Reshape(dims=(50,)))
        models.append(model_PersonalField24)

        model_PersonalField25 = Sequential()
        model_PersonalField25.add(Embedding(14, 50, input_length=1))
        model_PersonalField25.add(Reshape(dims=(50,)))
        models.append(model_PersonalField25)

        model_PersonalField26 = Sequential()
        model_PersonalField26.add(Embedding(14, 50, input_length=1))
        model_PersonalField26.add(Reshape(dims=(50,)))
        models.append(model_PersonalField26)

        model_PersonalField27 = Sequential()
        model_PersonalField27.add(Embedding(17, 50, input_length=1))
        model_PersonalField27.add(Reshape(dims=(50,)))
        models.append(model_PersonalField27)

        model_PersonalField28 = Sequential()
        model_PersonalField28.add(Embedding(7, 50, input_length=1))
        model_PersonalField28.add(Reshape(dims=(50,)))
        models.append(model_PersonalField28)

        model_PersonalField29 = Sequential()
        model_PersonalField29.add(Embedding(7, 50, input_length=1))
        model_PersonalField29.add(Reshape(dims=(50,)))
        models.append(model_PersonalField29)

        model_PersonalField30 = Sequential()
        model_PersonalField30.add(Embedding(12, 50, input_length=1))
        model_PersonalField30.add(Reshape(dims=(50,)))
        models.append(model_PersonalField30)

        model_PersonalField31 = Sequential()
        model_PersonalField31.add(Embedding(12, 50, input_length=1))
        model_PersonalField31.add(Reshape(dims=(50,)))
        models.append(model_PersonalField31)

        model_PersonalField32 = Sequential()
        model_PersonalField32.add(Embedding(12, 50, input_length=1))
        model_PersonalField32.add(Reshape(dims=(50,)))
        models.append(model_PersonalField32)

        model_PersonalField33 = Sequential()
        model_PersonalField33.add(Embedding(12, 50, input_length=1))
        model_PersonalField33.add(Reshape(dims=(50,)))
        models.append(model_PersonalField33)

        model_PersonalField34 = Sequential()
        model_PersonalField34.add(Embedding(7, 50, input_length=1))
        model_PersonalField34.add(Reshape(dims=(50,)))
        models.append(model_PersonalField34)

        model_PersonalField35 = Sequential()
        model_PersonalField35.add(Embedding(6, 50, input_length=1))
        model_PersonalField35.add(Reshape(dims=(50,)))
        models.append(model_PersonalField35)

        model_PersonalField36 = Sequential()
        model_PersonalField36.add(Embedding(6, 50, input_length=1))
        model_PersonalField36.add(Reshape(dims=(50,)))
        models.append(model_PersonalField36)

        model_PersonalField37 = Sequential()
        model_PersonalField37.add(Embedding(6, 50, input_length=1))
        model_PersonalField37.add(Reshape(dims=(50,)))
        models.append(model_PersonalField37)

        model_PersonalField38 = Sequential()
        model_PersonalField38.add(Embedding(6, 50, input_length=1))
        model_PersonalField38.add(Reshape(dims=(50,)))
        models.append(model_PersonalField38)

        model_PersonalField39 = Sequential()
        model_PersonalField39.add(Embedding(9, 50, input_length=1))
        model_PersonalField39.add(Reshape(dims=(50,)))
        models.append(model_PersonalField39)

        model_PersonalField40 = Sequential()
        model_PersonalField40.add(Embedding(10, 50, input_length=1))
        model_PersonalField40.add(Reshape(dims=(50,)))
        models.append(model_PersonalField40)

        model_PersonalField41 = Sequential()
        model_PersonalField41.add(Embedding(10, 50, input_length=1))
        model_PersonalField41.add(Reshape(dims=(50,)))
        models.append(model_PersonalField41)

        model_PersonalField42 = Sequential()
        model_PersonalField42.add(Embedding(10, 50, input_length=1))
        model_PersonalField42.add(Reshape(dims=(50,)))
        models.append(model_PersonalField42)

        model_PersonalField43 = Sequential()
        model_PersonalField43.add(Embedding(7, 50, input_length=1))
        model_PersonalField43.add(Reshape(dims=(50,)))
        models.append(model_PersonalField43)

        model_PersonalField44 = Sequential()
        model_PersonalField44.add(Embedding(12, 50, input_length=1))
        model_PersonalField44.add(Reshape(dims=(50,)))
        models.append(model_PersonalField44)

        model_PersonalField45 = Sequential()
        model_PersonalField45.add(Embedding(12, 50, input_length=1))
        model_PersonalField45.add(Reshape(dims=(50,)))
        models.append(model_PersonalField45)

        model_PersonalField46 = Sequential()
        model_PersonalField46.add(Embedding(12, 50, input_length=1))
        model_PersonalField46.add(Reshape(dims=(50,)))
        models.append(model_PersonalField46)

        model_PersonalField47 = Sequential()
        model_PersonalField47.add(Embedding(12, 50, input_length=1))
        model_PersonalField47.add(Reshape(dims=(50,)))
        models.append(model_PersonalField47)

        model_PersonalField48 = Sequential()
        model_PersonalField48.add(Embedding(7, 50, input_length=1))
        model_PersonalField48.add(Reshape(dims=(50,)))
        models.append(model_PersonalField48)

        model_PersonalField49 = Sequential()
        model_PersonalField49.add(Embedding(7, 50, input_length=1))
        model_PersonalField49.add(Reshape(dims=(50,)))
        models.append(model_PersonalField49)

        model_PersonalField50 = Sequential()
        model_PersonalField50.add(Embedding(7, 50, input_length=1))
        model_PersonalField50.add(Reshape(dims=(50,)))
        models.append(model_PersonalField50)

        model_PersonalField51 = Sequential()
        model_PersonalField51.add(Embedding(7, 50, input_length=1))
        model_PersonalField51.add(Reshape(dims=(50,)))
        models.append(model_PersonalField51)

        model_PersonalField52 = Sequential()
        model_PersonalField52.add(Embedding(7, 50, input_length=1))
        model_PersonalField52.add(Reshape(dims=(50,)))
        models.append(model_PersonalField52)

        model_PersonalField53 = Sequential()
        model_PersonalField53.add(Embedding(7, 50, input_length=1))
        model_PersonalField53.add(Reshape(dims=(50,)))
        models.append(model_PersonalField53)

        model_PersonalField54 = Sequential()
        model_PersonalField54.add(Embedding(11, 50, input_length=1))
        model_PersonalField54.add(Reshape(dims=(50,)))
        models.append(model_PersonalField54)

        model_PersonalField55 = Sequential()
        model_PersonalField55.add(Embedding(11, 50, input_length=1))
        model_PersonalField55.add(Reshape(dims=(50,)))
        models.append(model_PersonalField55)

        model_PersonalField56 = Sequential()
        model_PersonalField56.add(Embedding(12, 50, input_length=1))
        model_PersonalField56.add(Reshape(dims=(50,)))
        models.append(model_PersonalField56)

        model_PersonalField57 = Sequential()
        model_PersonalField57.add(Embedding(13, 50, input_length=1))
        model_PersonalField57.add(Reshape(dims=(50,)))
        models.append(model_PersonalField57)

        model_PersonalField58 = Sequential()
        model_PersonalField58.add(Embedding(7, 50, input_length=1))
        model_PersonalField58.add(Reshape(dims=(50,)))
        models.append(model_PersonalField58)

        model_PersonalField59 = Sequential()
        model_PersonalField59.add(Embedding(7, 50, input_length=1))
        model_PersonalField59.add(Reshape(dims=(50,)))
        models.append(model_PersonalField59)

        model_PersonalField60 = Sequential()
        model_PersonalField60.add(Embedding(7, 50, input_length=1))
        model_PersonalField60.add(Reshape(dims=(50,)))
        models.append(model_PersonalField60)

        model_PersonalField60 = Sequential()
        model_PersonalField60.add(Embedding(7, 50, input_length=1))
        model_PersonalField60.add(Reshape(dims=(50,)))
        models.append(model_PersonalField60)

        model_PersonalField61 = Sequential()
        model_PersonalField61.add(Embedding(7, 50, input_length=1))
        model_PersonalField61.add(Reshape(dims=(50,)))
        models.append(model_PersonalField61)

        model_PersonalField62 = Sequential()
        model_PersonalField62.add(Embedding(7, 50, input_length=1))
        model_PersonalField62.add(Reshape(dims=(50,)))
        models.append(model_PersonalField62)

        model_PersonalField63 = Sequential()
        model_PersonalField63.add(Embedding(7, 50, input_length=1))
        model_PersonalField63.add(Reshape(dims=(50,)))
        models.append(model_PersonalField63)

        model_PersonalField64 = Sequential()
        model_PersonalField64.add(Embedding(3, 50, input_length=1))
        model_PersonalField64.add(Reshape(dims=(50,)))
        models.append(model_PersonalField64)

        model_PersonalField65 = Sequential()
        model_PersonalField65.add(Embedding(3, 50, input_length=1))
        model_PersonalField65.add(Reshape(dims=(50,)))
        models.append(model_PersonalField65)

        model_PersonalField66 = Sequential()
        model_PersonalField66.add(Embedding(3, 50, input_length=1))
        model_PersonalField66.add(Reshape(dims=(50,)))
        models.append(model_PersonalField66)

        model_PersonalField67 = Sequential()
        model_PersonalField67.add(Embedding(3, 50, input_length=1))
        model_PersonalField67.add(Reshape(dims=(50,)))
        models.append(model_PersonalField67)

        model_PersonalField68 = Sequential()
        model_PersonalField68.add(Embedding(4, 50, input_length=1))
        model_PersonalField68.add(Reshape(dims=(50,)))
        models.append(model_PersonalField68)

        model_PersonalField69 = Sequential()
        model_PersonalField69.add(Embedding(6, 50, input_length=1))
        model_PersonalField69.add(Reshape(dims=(50,)))
        models.append(model_PersonalField69)

        model_PersonalField70 = Sequential()
        model_PersonalField70.add(Embedding(7, 50, input_length=1))
        model_PersonalField70.add(Reshape(dims=(50,)))
        models.append(model_PersonalField70)

        model_PersonalField71 = Sequential()
        model_PersonalField71.add(Embedding(7, 50, input_length=1))
        model_PersonalField71.add(Reshape(dims=(50,)))
        models.append(model_PersonalField71)

        model_PersonalField72 = Sequential()
        model_PersonalField72.add(Embedding(8, 50, input_length=1))
        model_PersonalField72.add(Reshape(dims=(50,)))
        models.append(model_PersonalField72)

        model_PersonalField73 = Sequential()
        model_PersonalField73.add(Embedding(7, 50, input_length=1))
        model_PersonalField73.add(Reshape(dims=(50,)))
        models.append(model_PersonalField73)

        model_PersonalField74 = Sequential()
        model_PersonalField74.add(Embedding(8, 50, input_length=1))
        model_PersonalField74.add(Reshape(dims=(50,)))
        models.append(model_PersonalField74)

        model_PersonalField75 = Sequential()
        model_PersonalField75.add(Embedding(9, 50, input_length=1))
        model_PersonalField75.add(Reshape(dims=(50,)))
        models.append(model_PersonalField75)

        model_PersonalField76 = Sequential()
        model_PersonalField76.add(Embedding(10, 50, input_length=1))
        model_PersonalField76.add(Reshape(dims=(50,)))
        models.append(model_PersonalField76)

        model_PersonalField77 = Sequential()
        model_PersonalField77.add(Embedding(11, 50, input_length=1))
        model_PersonalField77.add(Reshape(dims=(50,)))
        models.append(model_PersonalField77)

        model_PersonalField78 = Sequential()
        model_PersonalField78.add(Embedding(7, 50, input_length=1))
        model_PersonalField78.add(Reshape(dims=(50,)))
        models.append(model_PersonalField78)

        model_PersonalField79 = Sequential()
        model_PersonalField79.add(Embedding(13, 50, input_length=1))
        model_PersonalField79.add(Reshape(dims=(50,)))
        models.append(model_PersonalField79)

        model_PersonalField80 = Sequential()
        model_PersonalField80.add(Embedding(14, 50, input_length=1))
        model_PersonalField80.add(Reshape(dims=(50,)))
        models.append(model_PersonalField80)

        model_PersonalField81 = Sequential()
        model_PersonalField81.add(Embedding(14, 50, input_length=1))
        model_PersonalField81.add(Reshape(dims=(50,)))
        models.append(model_PersonalField81)

        model_PersonalField82 = Sequential()
        model_PersonalField82.add(Embedding(14, 50, input_length=1))
        model_PersonalField82.add(Reshape(dims=(50,)))
        models.append(model_PersonalField82)

        model_PersonalField83 = Sequential()
        model_PersonalField83.add(Embedding(7, 50, input_length=1))
        model_PersonalField83.add(Reshape(dims=(50,)))
        models.append(model_PersonalField83)

        model_PersonalField84 = Sequential()
        model_PersonalField84.add(Embedding(8, 50, input_length=1))
        model_PersonalField84.add(Reshape(dims=(50,)))
        models.append(model_PersonalField84)

        model_PropertyField1A = Sequential()
        model_PropertyField1A.add(Embedding(26, 50, input_length=1))
        model_PropertyField1A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField1A)

        model_PropertyField1B = Sequential()
        model_PropertyField1B.add(Embedding(26, 50, input_length=1))
        model_PropertyField1B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField1B)

        model_PropertyField2A = Sequential()
        model_PropertyField2A.add(Embedding(2, 50, input_length=1))
        model_PropertyField2A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField2A)

        model_PropertyField2B = Sequential()
        model_PropertyField2B.add(Embedding(21, 50, input_length=1))
        model_PropertyField2B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField2B)

        model_PropertyField3 = Sequential()
        model_PropertyField3.add(Embedding(3, 50, input_length=1))
        model_PropertyField3.add(Reshape(dims=(50,)))
        models.append(model_PropertyField3)

        model_PropertyField4 = Sequential()
        model_PropertyField4.add(Embedding(3, 50, input_length=1))
        model_PropertyField4.add(Reshape(dims=(50,)))
        models.append(model_PropertyField4)

        model_PropertyField5 = Sequential()
        model_PropertyField5.add(Embedding(2, 50, input_length=1))
        model_PropertyField5.add(Reshape(dims=(50,)))
        models.append(model_PropertyField5)

        model_PropertyField6 = Sequential()
        model_PropertyField6.add(Embedding(1, 50, input_length=1))
        model_PropertyField6.add(Reshape(dims=(50,)))
        models.append(model_PropertyField6)

        model_PropertyField7 = Sequential()
        model_PropertyField7.add(Embedding(19, 50, input_length=1))
        model_PropertyField7.add(Reshape(dims=(50,)))
        models.append(model_PropertyField7)

        model_PropertyField8 = Sequential()
        model_PropertyField8.add(Embedding(2, 50, input_length=1))
        model_PropertyField8.add(Reshape(dims=(50,)))
        models.append(model_PropertyField8)

        model_PropertyField9 = Sequential()
        model_PropertyField9.add(Embedding(3, 50, input_length=1))
        model_PropertyField9.add(Reshape(dims=(50,)))
        models.append(model_PropertyField9)

        model_PropertyField10 = Sequential()
        model_PropertyField10.add(Embedding(5, 50, input_length=1))
        model_PropertyField10.add(Reshape(dims=(50,)))
        models.append(model_PropertyField10)

        model_PropertyField11A = Sequential()
        model_PropertyField11A.add(Embedding(2, 50, input_length=1))
        model_PropertyField11A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField11A)

        model_PropertyField11B = Sequential()
        model_PropertyField11B.add(Embedding(5, 50, input_length=1))
        model_PropertyField11B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField11B)

        model_PropertyField12 = Sequential()
        model_PropertyField12.add(Embedding(7, 50, input_length=1))
        model_PropertyField12.add(Reshape(dims=(50,)))
        models.append(model_PropertyField12)

        model_PropertyField13 = Sequential()
        model_PropertyField13.add(Embedding(4, 50, input_length=1))
        model_PropertyField13.add(Reshape(dims=(50,)))
        models.append(model_PropertyField13)

        model_PropertyField14 = Sequential()
        model_PropertyField14.add(Embedding(4, 50, input_length=1))
        model_PropertyField14.add(Reshape(dims=(50,)))
        models.append(model_PropertyField14)

        model_PropertyField15 = Sequential()
        model_PropertyField15.add(Embedding(15, 50, input_length=1))
        model_PropertyField15.add(Reshape(dims=(50,)))
        models.append(model_PropertyField15)

        model_PropertyField16A = Sequential()
        model_PropertyField16A.add(Embedding(26, 50, input_length=1))
        model_PropertyField16A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField16A)

        model_PropertyField16B = Sequential()
        model_PropertyField16B.add(Embedding(26, 50, input_length=1))
        model_PropertyField16B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField16B)

        model_PropertyField17 = Sequential()
        model_PropertyField17.add(Embedding(8, 50, input_length=1))
        model_PropertyField17.add(Reshape(dims=(50,)))
        models.append(model_PropertyField17)

        model_PropertyField18 = Sequential()
        model_PropertyField18.add(Embedding(10, 50, input_length=1))
        model_PropertyField18.add(Reshape(dims=(50,)))
        models.append(model_PropertyField18)

        model_PropertyField19 = Sequential()
        model_PropertyField19.add(Embedding(10, 50, input_length=1))
        model_PropertyField19.add(Reshape(dims=(50,)))
        models.append(model_PropertyField19)

        model_PropertyField20 = Sequential()
        model_PropertyField20.add(Embedding(3, 50, input_length=1))
        model_PropertyField20.add(Reshape(dims=(50,)))
        models.append(model_PropertyField20)

        model_PropertyField21A = Sequential()
        model_PropertyField21A.add(Embedding(26, 50, input_length=1))
        model_PropertyField21A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField21A)

        model_PropertyField21B = Sequential()
        model_PropertyField21B.add(Embedding(26, 50, input_length=1))
        model_PropertyField21B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField21B)

        model_PropertyField22 = Sequential()
        model_PropertyField22.add(Embedding(5, 50, input_length=1))
        model_PropertyField22.add(Reshape(dims=(50,)))
        models.append(model_PropertyField22)

        model_PropertyField23 = Sequential()
        model_PropertyField23.add(Embedding(13, 50, input_length=1))
        model_PropertyField23.add(Reshape(dims=(50,)))
        models.append(model_PropertyField23)

        model_PropertyField24A = Sequential()
        model_PropertyField24A.add(Embedding(26, 50, input_length=1))
        model_PropertyField24A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField24A)

        model_PropertyField24B = Sequential()
        model_PropertyField24B.add(Embedding(26, 50, input_length=1))
        model_PropertyField24B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField24B)

        model_PropertyField25 = Sequential()
        model_PropertyField25.add(Embedding(11, 50, input_length=1))
        model_PropertyField25.add(Reshape(dims=(50,)))
        models.append(model_PropertyField25)

        model_PropertyField26A = Sequential()
        model_PropertyField26A.add(Embedding(26, 50, input_length=1))
        model_PropertyField26A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField26A)

        model_PropertyField26B = Sequential()
        model_PropertyField26B.add(Embedding(26, 50, input_length=1))
        model_PropertyField26B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField26B)

        model_PropertyField27 = Sequential()
        model_PropertyField27.add(Embedding(17, 50, input_length=1))
        model_PropertyField27.add(Reshape(dims=(50,)))
        models.append(model_PropertyField27)

        model_PropertyField28 = Sequential()
        model_PropertyField28.add(Embedding(4, 50, input_length=1))
        model_PropertyField28.add(Reshape(dims=(50,)))
        models.append(model_PropertyField28)

        model_PropertyField29 = Sequential()
        model_PropertyField29.add(Embedding(3, 50, input_length=1))
        model_PropertyField29.add(Reshape(dims=(50,)))
        models.append(model_PropertyField29)

        model_PropertyField30 = Sequential()
        model_PropertyField30.add(Embedding(2, 50, input_length=1))
        model_PropertyField30.add(Reshape(dims=(50,)))
        models.append(model_PropertyField30)

        model_PropertyField31 = Sequential()
        model_PropertyField31.add(Embedding(4, 50, input_length=1))
        model_PropertyField31.add(Reshape(dims=(50,)))
        models.append(model_PropertyField31)

        model_PropertyField32 = Sequential()
        model_PropertyField32.add(Embedding(3, 50, input_length=1))
        model_PropertyField32.add(Reshape(dims=(50,)))
        models.append(model_PropertyField32)

        model_PropertyField33 = Sequential()
        model_PropertyField33.add(Embedding(4, 50, input_length=1))
        model_PropertyField33.add(Reshape(dims=(50,)))
        models.append(model_PropertyField33)

        model_PropertyField34 = Sequential()
        model_PropertyField34.add(Embedding(3, 50, input_length=1))
        model_PropertyField34.add(Reshape(dims=(50,)))
        models.append(model_PropertyField34)

        model_PropertyField35 = Sequential()
        model_PropertyField35.add(Embedding(3, 50, input_length=1))
        model_PropertyField35.add(Reshape(dims=(50,)))
        models.append(model_PropertyField35)

        model_PropertyField36 = Sequential()
        model_PropertyField36.add(Embedding(3, 50, input_length=1))
        model_PropertyField36.add(Reshape(dims=(50,)))
        models.append(model_PropertyField36)

        model_PropertyField37 = Sequential()
        model_PropertyField37.add(Embedding(2, 50, input_length=1))
        model_PropertyField37.add(Reshape(dims=(50,)))
        models.append(model_PropertyField37)

        model_PropertyField38 = Sequential()
        model_PropertyField38.add(Embedding(3, 50, input_length=1))
        model_PropertyField38.add(Reshape(dims=(50,)))
        models.append(model_PropertyField38)

        model_PropertyField39A = Sequential()
        model_PropertyField39A.add(Embedding(26, 50, input_length=1))
        model_PropertyField39A.add(Reshape(dims=(50,)))
        models.append(model_PropertyField39A)

        model_PropertyField39B = Sequential()
        model_PropertyField39B.add(Embedding(26, 50, input_length=1))
        model_PropertyField39B.add(Reshape(dims=(50,)))
        models.append(model_PropertyField39B)

        model_GeographicField1A = Sequential()
        model_GeographicField1A.add(Embedding(26, 50, input_length=1))
        model_GeographicField1A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField1A)

        model_GeographicField1B = Sequential()
        model_GeographicField1B.add(Embedding(26, 50, input_length=1))
        model_GeographicField1B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField1B)

        model_GeographicField2A = Sequential()
        model_GeographicField2A.add(Embedding(26, 50, input_length=1))
        model_GeographicField2A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField2A)

        model_GeographicField2B = Sequential()
        model_GeographicField2B.add(Embedding(26, 50, input_length=1))
        model_GeographicField2B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField2B)

        model_GeographicField3A = Sequential()
        model_GeographicField3A.add(Embedding(25, 50, input_length=1))
        model_GeographicField3A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField3A)

        model_GeographicField3B = Sequential()
        model_GeographicField3B.add(Embedding(25, 50, input_length=1))
        model_GeographicField3B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField3B)

        model_GeographicField4A = Sequential()
        model_GeographicField4A.add(Embedding(25, 50, input_length=1))
        model_GeographicField4A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField4A)

        model_GeographicField4B = Sequential()
        model_GeographicField4B.add(Embedding(25, 50, input_length=1))
        model_GeographicField4B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField4B)

        model_GeographicField5A = Sequential()
        model_GeographicField5A.add(Embedding(2, 50, input_length=1))
        model_GeographicField5A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField5A)

        model_GeographicField5B = Sequential()
        model_GeographicField5B.add(Embedding(14, 50, input_length=1))
        model_GeographicField5B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField5B)

        model_GeographicField6A = Sequential()
        model_GeographicField6A.add(Embedding(26, 50, input_length=1))
        model_GeographicField6A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField6A)

        model_GeographicField6B = Sequential()
        model_GeographicField6B.add(Embedding(26, 50, input_length=1))
        model_GeographicField6B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField6B)

        model_GeographicField7A = Sequential()
        model_GeographicField7A.add(Embedding(25, 50, input_length=1))
        model_GeographicField7A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField7A)

        model_GeographicField7B = Sequential()
        model_GeographicField7B.add(Embedding(24, 50, input_length=1))
        model_GeographicField7B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField7B)

        model_GeographicField8A = Sequential()
        model_GeographicField8A.add(Embedding(26, 50, input_length=1))
        model_GeographicField8A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField8A)

        model_GeographicField8B = Sequential()
        model_GeographicField8B.add(Embedding(25, 50, input_length=1))
        model_GeographicField8B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField8B)

        model_GeographicField9A = Sequential()
        model_GeographicField9A.add(Embedding(26, 50, input_length=1))
        model_GeographicField9A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField9A)

        model_GeographicField9B = Sequential()
        model_GeographicField9B.add(Embedding(26, 50, input_length=1))
        model_GeographicField9B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField9B)

        model_GeographicField10A = Sequential()
        model_GeographicField10A.add(Embedding(1, 50, input_length=1))
        model_GeographicField10A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField10A)

        model_GeographicField10B = Sequential()
        model_GeographicField10B.add(Embedding(2, 50, input_length=1))
        model_GeographicField10B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField10B)

        model_GeographicField11A = Sequential()
        model_GeographicField11A.add(Embedding(26, 50, input_length=1))
        model_GeographicField11A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField11A)

        model_GeographicField11B = Sequential()
        model_GeographicField11B.add(Embedding(25, 50, input_length=1))
        model_GeographicField11B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField11B)

        model_GeographicField12A = Sequential()
        model_GeographicField12A.add(Embedding(26, 50, input_length=1))
        model_GeographicField12A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField12A)

        model_GeographicField12B = Sequential()
        model_GeographicField12B.add(Embedding(24, 50, input_length=1))
        model_GeographicField12B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField12B)

        model_GeographicField13A = Sequential()
        model_GeographicField13A.add(Embedding(25, 50, input_length=1))
        model_GeographicField13A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField13A)

        model_GeographicField13B = Sequential()
        model_GeographicField13B.add(Embedding(25, 50, input_length=1))
        model_GeographicField13B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField13B)

        model_GeographicField14A = Sequential()
        model_GeographicField14A.add(Embedding(2, 50, input_length=1))
        model_GeographicField14A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField14A)

        model_GeographicField14B = Sequential()
        model_GeographicField14B.add(Embedding(20, 50, input_length=1))
        model_GeographicField14B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField14B)

        model_GeographicField15A = Sequential()
        model_GeographicField15A.add(Embedding(26, 50, input_length=1))
        model_GeographicField15A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField15A)

        model_GeographicField15B = Sequential()
        model_GeographicField15B.add(Embedding(24, 50, input_length=1))
        model_GeographicField15B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField15B)

        model_GeographicField16A = Sequential()
        model_GeographicField16A.add(Embedding(26, 50, input_length=1))
        model_GeographicField16A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField16A)

        model_GeographicField16B = Sequential()
        model_GeographicField16B.add(Embedding(26, 50, input_length=1))
        model_GeographicField16B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField16B)

        model_GeographicField17A = Sequential()
        model_GeographicField17A.add(Embedding(26, 50, input_length=1))
        model_GeographicField17A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField17A)

        model_GeographicField17B = Sequential()
        model_GeographicField17B.add(Embedding(26, 50, input_length=1))
        model_GeographicField17B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField17B)

        model_GeographicField18A = Sequential()
        model_GeographicField18A.add(Embedding(2, 50, input_length=1))
        model_GeographicField18A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField18A)

        model_GeographicField18B = Sequential()
        model_GeographicField18B.add(Embedding(25, 50, input_length=1))
        model_GeographicField18B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField18B)

        model_GeographicField19A = Sequential()
        model_GeographicField19A.add(Embedding(26, 50, input_length=1))
        model_GeographicField19A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField19A)

        model_GeographicField19B = Sequential()
        model_GeographicField19B.add(Embedding(26, 50, input_length=1))
        model_GeographicField19B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField19B)

        model_GeographicField20A = Sequential()
        model_GeographicField20A.add(Embedding(26, 50, input_length=1))
        model_GeographicField20A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField20A)

        model_GeographicField20B = Sequential()
        model_GeographicField20B.add(Embedding(26, 50, input_length=1))
        model_GeographicField20B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField20B)

        model_GeographicField21A = Sequential()
        model_GeographicField21A.add(Embedding(2, 50, input_length=1))
        model_GeographicField21A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField21A)

        model_GeographicField21B = Sequential()
        model_GeographicField21B.add(Embedding(23, 50, input_length=1))
        model_GeographicField21B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField21B)

        model_GeographicField22A = Sequential()
        model_GeographicField22A.add(Embedding(2, 50, input_length=1))
        model_GeographicField22A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField22A)

        model_GeographicField22B = Sequential()
        model_GeographicField22B.add(Embedding(12, 50, input_length=1))
        model_GeographicField22B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField22B)

        model_GeographicField23A = Sequential()
        model_GeographicField23A.add(Embedding(2, 50, input_length=1))
        model_GeographicField23A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField23A)

        model_GeographicField23B = Sequential()
        model_GeographicField23B.add(Embedding(26, 50, input_length=1))
        model_GeographicField23B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField23B)

        model_GeographicField24A = Sequential()
        model_GeographicField24A.add(Embedding(25, 50, input_length=1))
        model_GeographicField24A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField24A)

        model_GeographicField24B = Sequential()
        model_GeographicField24B.add(Embedding(25, 50, input_length=1))
        model_GeographicField24B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField24B)

        model_GeographicField25A = Sequential()
        model_GeographicField25A.add(Embedding(25, 50, input_length=1))
        model_GeographicField25A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField25A)

        model_GeographicField25B = Sequential()
        model_GeographicField25B.add(Embedding(25, 50, input_length=1))
        model_GeographicField25B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField25B)

        model_GeographicField26A = Sequential()
        model_GeographicField26A.add(Embedding(26, 50, input_length=1))
        model_GeographicField26A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField26A)

        model_GeographicField26B = Sequential()
        model_GeographicField26B.add(Embedding(25, 50, input_length=1))
        model_GeographicField26B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField26B)

        model_GeographicField27A = Sequential()
        model_GeographicField27A.add(Embedding(25, 50, input_length=1))
        model_GeographicField27A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField27A)

        model_GeographicField27B = Sequential()
        model_GeographicField27B.add(Embedding(24, 50, input_length=1))
        model_GeographicField27B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField27B)

        model_GeographicField28A = Sequential()
        model_GeographicField28A.add(Embedding(26, 50, input_length=1))
        model_GeographicField28A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField28A)

        model_GeographicField28B = Sequential()
        model_GeographicField28B.add(Embedding(26, 50, input_length=1))
        model_GeographicField28B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField28B)

        model_GeographicField29A = Sequential()
        model_GeographicField29A.add(Embedding(26, 50, input_length=1))
        model_GeographicField29A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField29A)

        model_GeographicField29B = Sequential()
        model_GeographicField29B.add(Embedding(26, 50, input_length=1))
        model_GeographicField29B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField29B)

        model_GeographicField30A = Sequential()
        model_GeographicField30A.add(Embedding(26, 50, input_length=1))
        model_GeographicField30A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField30A)

        model_GeographicField30B = Sequential()
        model_GeographicField30B.add(Embedding(26, 50, input_length=1))
        model_GeographicField30B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField30B)

        model_GeographicField31A = Sequential()
        model_GeographicField31A.add(Embedding(26, 50, input_length=1))
        model_GeographicField31A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField31A)

        model_GeographicField31B = Sequential()
        model_GeographicField31B.add(Embedding(26, 50, input_length=1))
        model_GeographicField31B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField31B)

        model_GeographicField32A = Sequential()
        model_GeographicField32A.add(Embedding(26, 50, input_length=1))
        model_GeographicField32A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField32A)

        model_GeographicField32B = Sequential()
        model_GeographicField32B.add(Embedding(26, 50, input_length=1))
        model_GeographicField32B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField32B)

        model_GeographicField33A = Sequential()
        model_GeographicField33A.add(Embedding(26, 50, input_length=1))
        model_GeographicField33A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField33A)

        model_GeographicField33B = Sequential()
        model_GeographicField33B.add(Embedding(26, 50, input_length=1))
        model_GeographicField33B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField33B)

        model_GeographicField34A = Sequential()
        model_GeographicField34A.add(Embedding(26, 50, input_length=1))
        model_GeographicField34A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField34A)

        model_GeographicField34B = Sequential()
        model_GeographicField34B.add(Embedding(26, 50, input_length=1))
        model_GeographicField34B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField34B)

        model_GeographicField35A = Sequential()
        model_GeographicField35A.add(Embedding(26, 50, input_length=1))
        model_GeographicField35A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField35A)

        model_GeographicField35B = Sequential()
        model_GeographicField35B.add(Embedding(26, 50, input_length=1))
        model_GeographicField35B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField35B)

        model_GeographicField36A = Sequential()
        model_GeographicField36A.add(Embedding(26, 50, input_length=1))
        model_GeographicField36A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField36A)

        model_GeographicField36B = Sequential()
        model_GeographicField36B.add(Embedding(26, 50, input_length=1))
        model_GeographicField36B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField36B)

        model_GeographicField37A = Sequential()
        model_GeographicField37A.add(Embedding(26, 50, input_length=1))
        model_GeographicField37A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField37A)

        model_GeographicField37B = Sequential()
        model_GeographicField37B.add(Embedding(26, 50, input_length=1))
        model_GeographicField37B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField37B)

        model_GeographicField38A = Sequential()
        model_GeographicField38A.add(Embedding(26, 50, input_length=1))
        model_GeographicField38A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField38A)

        model_GeographicField38B = Sequential()
        model_GeographicField38B.add(Embedding(26, 50, input_length=1))
        model_GeographicField38B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField38B)

        model_GeographicField39A = Sequential()
        model_GeographicField39A.add(Embedding(26, 50, input_length=1))
        model_GeographicField39A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField39A)

        model_GeographicField39B = Sequential()
        model_GeographicField39B.add(Embedding(26, 50, input_length=1))
        model_GeographicField39B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField39B)

        model_GeographicField40A = Sequential()
        model_GeographicField40A.add(Embedding(26, 50, input_length=1))
        model_GeographicField40A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField40A)

        model_GeographicField40A = Sequential()
        model_GeographicField40A.add(Embedding(26, 50, input_length=1))
        model_GeographicField40A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField40A)

        model_GeographicField40B = Sequential()
        model_GeographicField40B.add(Embedding(26, 50, input_length=1))
        model_GeographicField40B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField40B)

        model_GeographicField41A = Sequential()
        model_GeographicField41A.add(Embedding(26, 50, input_length=1))
        model_GeographicField41A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField41A)

        model_GeographicField41B = Sequential()
        model_GeographicField41B.add(Embedding(26, 50, input_length=1))
        model_GeographicField41B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField41B)

        model_GeographicField42A = Sequential()
        model_GeographicField42A.add(Embedding(26, 50, input_length=1))
        model_GeographicField42A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField42A)

        model_GeographicField42B = Sequential()
        model_GeographicField42B.add(Embedding(26, 50, input_length=1))
        model_GeographicField42B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField42B)

        model_GeographicField43A = Sequential()
        model_GeographicField43A.add(Embedding(26, 50, input_length=1))
        model_GeographicField43A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField43A)

        model_GeographicField43B = Sequential()
        model_GeographicField43B.add(Embedding(26, 50, input_length=1))
        model_GeographicField43B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField43B)

        model_GeographicField44A = Sequential()
        model_GeographicField44A.add(Embedding(26, 50, input_length=1))
        model_GeographicField44A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField44A)

        model_GeographicField44B = Sequential()
        model_GeographicField44B.add(Embedding(26, 50, input_length=1))
        model_GeographicField44B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField44B)

        model_GeographicField45A = Sequential()
        model_GeographicField45A.add(Embedding(26, 50, input_length=1))
        model_GeographicField45A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField45A)

        model_GeographicField45B = Sequential()
        model_GeographicField45B.add(Embedding(26, 50, input_length=1))
        model_GeographicField45B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField45B)

        model_GeographicField46A = Sequential()
        model_GeographicField46A.add(Embedding(26, 50, input_length=1))
        model_GeographicField46A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField46A)

        model_GeographicField46B = Sequential()
        model_GeographicField46B.add(Embedding(26, 50, input_length=1))
        model_GeographicField46B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField46B)

        model_GeographicField47A = Sequential()
        model_GeographicField47A.add(Embedding(26, 50, input_length=1))
        model_GeographicField47A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField47A)

        model_GeographicField47B = Sequential()
        model_GeographicField47B.add(Embedding(25, 50, input_length=1))
        model_GeographicField47B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField47B)

        model_GeographicField48A = Sequential()
        model_GeographicField48A.add(Embedding(26, 50, input_length=1))
        model_GeographicField48A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField48A)

        model_GeographicField48B = Sequential()
        model_GeographicField48B.add(Embedding(26, 50, input_length=1))
        model_GeographicField48B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField48B)

        model_GeographicField49A = Sequential()
        model_GeographicField49A.add(Embedding(26, 50, input_length=1))
        model_GeographicField49A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField49A)

        model_GeographicField49B = Sequential()
        model_GeographicField49B.add(Embedding(26, 50, input_length=1))
        model_GeographicField49B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField49B)

        model_GeographicField50A = Sequential()
        model_GeographicField50A.add(Embedding(26, 50, input_length=1))
        model_GeographicField50A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField50A)

        model_GeographicField50B = Sequential()
        model_GeographicField50B.add(Embedding(26, 50, input_length=1))
        model_GeographicField50B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField50B)

        model_GeographicField51A = Sequential()
        model_GeographicField51A.add(Embedding(26, 50, input_length=1))
        model_GeographicField51A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField51A)

        model_GeographicField51B = Sequential()
        model_GeographicField51B.add(Embedding(26, 50, input_length=1))
        model_GeographicField51B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField51B)

        model_GeographicField52A = Sequential()
        model_GeographicField52A.add(Embedding(26, 50, input_length=1))
        model_GeographicField52A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField52A)

        model_GeographicField52B = Sequential()
        model_GeographicField52B.add(Embedding(26, 50, input_length=1))
        model_GeographicField52B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField52B)

        model_GeographicField53A = Sequential()
        model_GeographicField53A.add(Embedding(26, 50, input_length=1))
        model_GeographicField53A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField53A)

        model_GeographicField53B = Sequential()
        model_GeographicField53B.add(Embedding(26, 50, input_length=1))
        model_GeographicField53B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField53B)

        model_GeographicField54A = Sequential()
        model_GeographicField54A.add(Embedding(26, 50, input_length=1))
        model_GeographicField54A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField54A)

        model_GeographicField54B = Sequential()
        model_GeographicField54B.add(Embedding(26, 50, input_length=1))
        model_GeographicField54B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField54B)

        model_GeographicField55A = Sequential()
        model_GeographicField55A.add(Embedding(26, 50, input_length=1))
        model_GeographicField55A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField55A)

        model_GeographicField55B = Sequential()
        model_GeographicField55B.add(Embedding(26, 50, input_length=1))
        model_GeographicField55B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField55B)

        model_GeographicField56A = Sequential()
        model_GeographicField56A.add(Embedding(2, 50, input_length=1))
        model_GeographicField56A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField56A)

        model_GeographicField56B = Sequential()
        model_GeographicField56B.add(Embedding(25, 50, input_length=1))
        model_GeographicField56B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField56B)

        model_GeographicField57A = Sequential()
        model_GeographicField57A.add(Embedding(26, 50, input_length=1))
        model_GeographicField57A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField57A)

        model_GeographicField57B = Sequential()
        model_GeographicField57B.add(Embedding(26, 50, input_length=1))
        model_GeographicField57B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField57B)

        model_GeographicField58A = Sequential()
        model_GeographicField58A.add(Embedding(26, 50, input_length=1))
        model_GeographicField58A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField58A)

        model_GeographicField58B = Sequential()
        model_GeographicField58B.add(Embedding(26, 50, input_length=1))
        model_GeographicField58B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField58B)

        model_GeographicField59A = Sequential()
        model_GeographicField59A.add(Embedding(26, 50, input_length=1))
        model_GeographicField59A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField59A)

        model_GeographicField59B = Sequential()
        model_GeographicField59B.add(Embedding(26, 50, input_length=1))
        model_GeographicField59B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField59B)

        model_GeographicField60A = Sequential()
        model_GeographicField60A.add(Embedding(2, 50, input_length=1))
        model_GeographicField60A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField60A)

        model_GeographicField60B = Sequential()
        model_GeographicField60B.add(Embedding(26, 50, input_length=1))
        model_GeographicField60B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField60B)

        model_GeographicField61A = Sequential()
        model_GeographicField61A.add(Embedding(2, 50, input_length=1))
        model_GeographicField61A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField61A)

        model_GeographicField61B = Sequential()
        model_GeographicField61B.add(Embedding(25, 50, input_length=1))
        model_GeographicField61B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField61B)

        model_GeographicField62A = Sequential()
        model_GeographicField62A.add(Embedding(2, 50, input_length=1))
        model_GeographicField62A.add(Reshape(dims=(50,)))
        models.append(model_GeographicField62A)

        model_GeographicField62B = Sequential()
        model_GeographicField62B.add(Embedding(19, 50, input_length=1))
        model_GeographicField62B.add(Reshape(dims=(50,)))
        models.append(model_GeographicField62B)

        model_GeographicField63 = Sequential()
        model_GeographicField63.add(Embedding(3, 50, input_length=1))
        model_GeographicField63.add(Reshape(dims=(50,)))
        models.append(model_GeographicField63)

        model_GeographicField64 = Sequential()
        model_GeographicField64.add(Embedding(4, 50, input_length=1))
        model_GeographicField64.add(Reshape(dims=(50,)))
        models.append(model_GeographicField64)
        
        model_year = Sequential()
        model_year.add(Embedding(3, 50, input_length=1))
        model_year.add(Reshape(dims=(50,)))
        models.append(model_year)
        
        model_month = Sequential()
        model_month.add(Embedding(12, 50, input_length=1))
        model_month.add(Reshape(dims=(50,)))
        models.append(model_month)
        
        model_monthday = Sequential()
        model_monthday.add(Embedding(31, 50, input_length=1))
        model_monthday.add(Reshape(dims=(50,)))
        models.append(model_monthday)
        
        model_weekday = Sequential()
        model_weekday.add(Embedding(7, 50, input_length=1))
        model_weekday.add(Reshape(dims=(50,)))
        models.append(model_weekday)
        
        model_yearday = Sequential()
        model_yearday.add(Embedding(365, 50, input_length=1))
        model_yearday.add(Reshape(dims=(50,)))
        models.append(model_yearday)
        
        model_daysDurOrigQuote = Sequential()
        model_daysDurOrigQuote.add(Embedding(868, 50, input_length=1))
        model_daysDurOrigQuote.add(Reshape(dims=(50,)))
        models.append(model_daysDurOrigQuote)
        
        model_logDaysDurOrigQuote = Sequential()
        model_logDaysDurOrigQuote.add(Embedding(868, 50, input_length=1))
        model_logDaysDurOrigQuote.add(Reshape(dims=(50,)))
        models.append(model_logDaysDurOrigQuote)
        
        model_CoverageNeg1s = Sequential()
        model_CoverageNeg1s.add(Embedding(3, 50, input_length=1))
        model_CoverageNeg1s.add(Reshape(dims=(50,)))
        models.append(model_CoverageNeg1s)
        
        model_SalesNeg1s = Sequential()
        model_SalesNeg1s.add(Embedding(2, 50, input_length=1))
        model_SalesNeg1s.add(Reshape(dims=(50,)))
        models.append(model_SalesNeg1s)
        
        model_PropertyNeg1s = Sequential()
        model_PropertyNeg1s.add(Embedding(14, 50, input_length=1))
        model_PropertyNeg1s.add(Reshape(dims=(50,)))
        models.append(model_PropertyNeg1s)
        
        model_GeoNeg1s = Sequential()
        model_GeoNeg1s.add(Embedding(24, 50, input_length=1))
        model_GeoNeg1s.add(Reshape(dims=(50,)))
        models.append(model_GeoNeg1s)
        
        model_PersonalNeg1s = Sequential()
        model_PersonalNeg1s.add(Embedding(6, 50, input_length=1))
        model_PersonalNeg1s.add(Reshape(dims=(50,)))
        models.append(model_PersonalNeg1s)
        
        model_TotalNeg1s = Sequential()
        model_TotalNeg1s.add(Embedding(34, 50, input_length=1))
        model_TotalNeg1s.add(Reshape(dims=(50,)))
        models.append(model_TotalNeg1s)
        
        model_QuoteConversion_Flag = Sequential()
        model_QuoteConversion_Flag.add(Embedding(2, 50, input_length=1))
        model_QuoteConversion_Flag.add(Reshape(dims=(50,)))
        models.append(model_QuoteConversion_Flag)



        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dropout(0.02))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self):
        if self.train_ratio < 1:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           validation_data=(self.preprocessing(self.X_val), self._val_for_fit(self.y_val)),
                           nb_epoch=self.nb_epoch, batch_size=128,
                           # callbacks=[self.checkpointer],
                           )
            # self.model.load_weights('best_model_weights.hdf5')
            print("Result on validation data: ", self.evaluate())
        else:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           nb_epoch=self.nb_epoch, batch_size=128)

    def guess(self, feature):
        feature = numpy.array(feature).reshape(1, -1)
        return self._val_for_pred(self.model.predict(self.preprocessing(feature)))[0][0]