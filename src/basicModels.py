from model_blocks import *
from model_general import *
from generate_lstm import *
from generate_images import *
from keras.models import Model
from keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

def lstm_model(EXP_NO, data_type, forecasting_horizon, seq_len, spdDict,
               n_lstm_layers, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units,
               opt_learningrate, loss_func, max_epoch,
               n_train=50000, n_val=10000, n_test=10000, seed=821, temp_type=None, normalization_opt='raw'):

    (trainimage, trainY, trainY_raw, valimage, valY, valY_raw, testimage, testY, testY_raw) = generateTimeSeriesSet(data_type, forecasting_horizon, seq_len, spdDict, n_train, n_val, n_test, seed, temp_type)
    (trainLimits, valLimits, testLimits) = generateSpdLimits(data_type, normalization_opt, n_train=50000, n_val=10000, n_test=10000, seed=821)

    if normalization_opt == 'raw':
        normalized = False
    else:
        normalized = True

    input = Input(shape=(seq_len,), name='lstm_input')
    res = repeated_lstm(input, n_lstm_layers, seq_len, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units)
    res = Dense(units=1)(res)

    model = Model(inputs = [input], outputs=res)
    opt = Adam(lr=opt_learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

    model.compile(loss=loss_func, optimizer=opt)
    history = EpochHistory(valimage=valimage, valY=valY_raw, testimage=testimage, testY=testY_raw, valLimits=valLimits, testLimits=testLimits, log_filepath=str(EXP_NO), normalized=normalized)

    model.fit([trainimage], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1, callbacks=[history], verbose=0, shuffle=True)

    return

def cnnlstm_model(EXP_NO, data_type, img_size, input_depth, forecasting_horizon, seq_len, spdArray,
                  batch_size, batchnorm_on, dropout_on, n_conv_layers, conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                  n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                  n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt = 'all', temp_type=None, normalization_opt='raw'):

    (trainimage, trainY, trainY_raw, valimage, valY, valY_raw, testimage, testY, testY_raw) = generateImageset(
        data_type, img_size, forecasting_horizon, seq_len, spdArray, n_train, n_val, n_test, seed, traj_opt, normalization_opt, temp_type)
    (trainLimits, valLimits, testLimits) = generateSpdLimits(data_type, normalization_opt, n_train=50000, n_val=10000, n_test=10000, seed=821)

    if normalization_opt == 'raw':
        normalized = False
    else:
        normalized = True

    input = Input(shape=(seq_len, img_size, img_size, None), name='cnn_input')
    res = repeated_cnnlstm(input, input_depth, seq_len, batch_size, batchnorm_on, dropout_on, n_conv_layers,
                           conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                           n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units)
    res = Dense(units=1)(res)

    model = Model(inputs=[input], outputs=res)

    opt = Adam(lr=opt_learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

    model.compile(loss=loss_func, optimizer=opt)
    history = EpochHistory(valimage=valimage, valY=valY_raw, testimage=testimage, testY=testY_raw, valLimits=valLimits, testLimits=testLimits, log_filepath=str(EXP_NO), normalized=normalized)

    model.fit([trainimage], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1, callbacks=[history], verbose=0, shuffle=True)

    del trainimage
    del valimage
    del testimage

    return

def nd_lstm_model(EXP_NO, data_type, img_size, spdDict, forecasting_horizon, seq_len, spdArray, batch_size, dropout_on,
                  n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                  n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt = 'all', temp_type=None, normalization_opt='raw'):

    (trainimage, trainY, trainY_raw, valimage, valY, valY_raw, testimage, testY, testY_raw) = \
        generateNdVector(data_type, img_size, forecasting_horizon, seq_len, spdArray, spdDict, n_train, n_val, n_test, seed, traj_opt, normalization_opt, temp_type)

    (trainLimits, valLimits, testLimits) = generateSpdLimits(data_type, normalization_opt, n_train=50000, n_val=10000, n_test=10000, seed=821)

    if normalization_opt == 'raw':
        normalized = False
    else:
        normalized = True

    if data_type % 10 in [1, 2]:
        input_depth = 2
    else:
        input_depth = 5

    input = Input(shape=(seq_len, input_depth,), name='lstm_input')
    res = repeated_lstm(input, n_lstm_layers, seq_len, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units)
    res = Dense(units=1)(res)

    model = Model(inputs=[input], outputs=res)
    opt = Adam(lr=opt_learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)
    # print model.summary()

    model.compile(loss=loss_func, optimizer=opt)
    history = EpochHistory(valimage=valimage, valY=valY_raw, testimage=testimage, testY=testY_raw, valLimits=valLimits, testLimits=testLimits, log_filepath=str(EXP_NO), normalized=normalized)

    model.fit([trainimage], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1, callbacks=[history], verbose=0, shuffle=True)

    del trainimage
    del valimage
    del testimage

    return

def cnn3d_model(EXP_NO, data_type, img_size, forecasting_horizon, seq_len, spdArray, batch_size, batchnorm_on, dropout_on, n_conv_layers, conv1_depth, conv2_depth, conv3_depth,
                conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt='all', temp_type=None, normalization_opt='raw'):
    (trainimage, trainY, trainY_raw, valimage, valY, valY_raw, testimage, testY, testY_raw) = generateImageset(
        data_type, img_size, forecasting_horizon, seq_len, spdArray, n_train, n_val, n_test, seed, traj_opt, normalization_opt, temp_type)
    (trainLimits, valLimits, testLimits) = generateSpdLimits(data_type, normalization_opt, n_train=50000, n_val=10000, n_test=10000, seed=821)

    if normalization_opt == 'raw':
        normalized = False
    else:
        normalized = True

    if data_type % 10 in [1, 2]:
        input_depth = 1
    else:
        input_depth = 4

    input = Input(shape=(seq_len, img_size, img_size, input_depth), name='cnn_input')
    spatial = Conv3D(conv1_depth, kernel_size = (input_depth, conv_filter_size, conv_filter_size),
                     padding='same', strides=(1, 1, 1), kernel_initializer='glorot_uniform', activation=None)(input)
    if pooling_on == 1:
        spatial = MaxPooling3D(pool_size = (pooling_size, pooling_size, pooling_size))(spatial)
    spatial = Activation(act_conv)(spatial)
    if batchnorm_on == 1:
        spatial = BatchNormalization()(spatial)

    spatial = Conv3D(conv2_depth, kernel_size = (conv_filter_size, conv_filter_size, conv_filter_size),
                     padding='same', strides=(1, 1, 1), kernel_initializer='glorot_uniform', activation=None)(spatial)
    spatial = Activation(act_conv)(spatial)
    if batchnorm_on == 1:
        spatial = BatchNormalization()(spatial)

    spatial = Conv3D(conv3_depth, kernel_size=(conv_filter_size, conv_filter_size, conv_filter_size),
                     padding='same', strides=(1, 1, 1), kernel_initializer='glorot_uniform', activation=None)(spatial)
    spatial = Activation(act_conv)(spatial)
    if batchnorm_on == 1:
        spatial = BatchNormalization()(spatial)

    spatial = Flatten()(spatial)
    spatial_out = Dense(units=conv_fc_units)(spatial)

    if dropout_on == 1:
        res = Dropout(0.2)(spatial_out)
    else:
        res = spatial_out
    res = Dense(units = lstm_fc_units)(res)

    if n_fc_layers == 2:
        res = Dense(units=last_fc_units)(res)
    res = Dense(units=1)(res)

    model = Model(inputs=[input], outputs=res)

    opt = Adam(lr=opt_learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

    model.compile(loss=loss_func, optimizer=opt)
    history = EpochHistory(valimage=valimage, valY=valY_raw, testimage=testimage, testY=testY_raw, valLimits=valLimits, testLimits=testLimits,
                           log_filepath=str(EXP_NO), normalized=normalized)
    model.fit([trainimage], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1, callbacks=[history], verbose=0, shuffle=True)

    del trainimage
    del valimage
    del testimage

    return