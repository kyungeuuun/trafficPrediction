from keras.callbacks import Callback
from keras import backend as K
import pandas as pd
import tensorflow as tf
import time as tt
import numpy as np

def param_colnames():
    return ['forecasting_horizon', 'data_type', 'seq_len', 'img_size',
            'batch_size', 'max_epoch', 'optimizer', 'opt_learningrate', 'batchnorm_on', 'dropout_on', 'loss_func', 'earlystop_on',
            'n_conv_layers', 'conv1_depth', 'conv2_depth', 'conv3_depth', 'conv4_depth', 'conv5_depth', 'conv_fc_units', 'act_conv', 'pooling_on', 'pooling_size',
            'conv_filter_size', 'n_lstm_layers', 'lstm_units', 'act_lstm', 'lstm_fc_units', 'n_fc_layers', 'last_fc_units', 'temp_type', 'traj_opt', 'normalization_opt']

def perf_colnames():
    return ['epoch', 'train_mape', 'train_mae', 'train_rmse', 'val_mape', 'val_mae', 'val_rmse',
            'test_mape', 'test_mae', 'test_rmse', 'train_time']

def nanmean(x):
    return tf.reduce_mean(tf.boolean_mask(x, tf.logical_not(tf.is_inf(x))), axis = -1)

def mean_absolute_percentage_error(y_true, y_pred):
    return nanmean(K.abs(y_true - y_pred) / y_true)

def mean_absolute_error(y_true, y_pred):
    return nanmean(K.abs(y_true - y_pred))

def mean_root_squared_error(y_true, y_pred):
    return tf.sqrt(nanmean(K.square(y_true - y_pred)))

def perfMetrics(err, truth):
    mae = np.mean(np.abs(err))
    mape = np.true_divide(np.abs(err), truth)
    mape = mape[~np.isnan(mape)]
    mape = np.mean(mape)
    rmse = np.sqrt(np.mean(np.square(err)))

    return (mape, mae, rmse)

class EpochHistory(Callback):
    def __init__(self, valimage, valY, testimage, testY, valLimits, testLimits, log_filepath, normalized=True):
        self.valimage = valimage
        self.valY = valY
        self.testimage = testimage
        self.testY = testY
        self.log_filepath = log_filepath
        self.valLimits = valLimits
        self.testLimits = testLimits
        self.normalized = normalized

    def on_train_begin(self, logs={}):
        self.perf = pd.DataFrame(columns = ('Epoch', 'val_mape', 'val_mae', 'val_rmse', 'test_mape', 'test_mae', 'test_rmse', 'time'))
        self.st = tt.time()
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        if self.normalized == True:
            val_predicted = np.squeeze(self.model.predict(self.valimage)) * self.valLimits
            test_predicted = np.squeeze(self.model.predict(self.testimage)) * self.testLimits
        else:
            val_predicted = np.squeeze(self.model.predict(self.valimage))
            test_predicted = np.squeeze(self.model.predict(self.testimage))

        val = perfMetrics(val_predicted - self.valY, self.valY)
        test = perfMetrics(test_predicted - self.testY, self.testY)
        self.losses.append(logs.get('loss'))

        self.perf.loc[epoch] = [epoch, val[0], val[1], val[2], test[0], test[1], test[2], tt.time() - self.st]
        # print self.model.evaluate(self.trainimage, self.lossY, verbose=0) # loss..
        print('### EPOCH %i / TIME %.1f ### loss %.3f // validation %.3f %.3f %.3f // test %.3f %.3f %.3f' %(
            epoch, tt.time() - self.st, self.losses[-1], val[0], val[1], val[2], test[0], test[1], test[2]))

        return

    def on_train_end(self, logs={}):
        self.perf['loss'] = self.losses
        self.perf.to_csv('/home/keun/PycharmProjects/trafficPrediction/log/' + str(self.log_filepath) + '.csv')