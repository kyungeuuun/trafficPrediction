from keras import activations
from keras.engine.topology import Layer
from keras.layers import LSTM, InputLayer, Dense, Input, Flatten, concatenate, Reshape, MaxPooling2D, Activation, Dropout, MaxPooling3D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf

class Local_Seq_Conv(Layer):
    def __init__(self, output_dim, seq_len, kernel_size, activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', padding='same', strides=(1, 1), **kwargs):
        super(Local_Seq_Conv, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bias_initializer = bias_initializer
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel = []
        self.bias = []
        for eachlen in range(self.seq_len):
            self.kernel += [self.add_weight(shape=self.kernel_size, initializer=self.kernel_initializer, trainable=True, name='kernel_{0}'.format(eachlen))]
            self.bias += [self.add_weight(shape=(self.kernel_size[-1],), initializer=self.bias_initializer, trainable=True, name='bias_{0}'.format(eachlen))]
        self.build = True

    def call(self, inputs):
        output = []
        for eachlen in range(self.seq_len):
            tmp = K.bias_add(K.conv2d(inputs[:, eachlen, :, :, :], self.kernel[eachlen], strides=self.strides, padding=self.padding), self.bias[eachlen])
            if self.activation is not None:
                output += [self.activation(tmp)]
        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)


def repeated_cnnlstm(input, input_depth, seq_len, batch_size, batchnorm_on, dropout_on, n_conv_layers,
                     conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                     n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units):

    spatial = Local_Seq_Conv(output_dim=conv1_depth, seq_len=seq_len, kernel_size=(conv_filter_size, conv_filter_size, input_depth, conv1_depth),
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(input)
    if pooling_on == 1:
        spatial = MaxPooling3D(pool_size=(1, pooling_size, pooling_size))(spatial)
    spatial = Activation(act_conv)(spatial)
    if batchnorm_on == 1:
        spatial = BatchNormalization()(spatial)

    spatial = Local_Seq_Conv(output_dim=conv2_depth, seq_len=seq_len,
                             kernel_size=(conv_filter_size, conv_filter_size, conv1_depth, conv2_depth), kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    spatial = Activation(act_conv)(spatial)
    if batchnorm_on == 1:
        spatial = BatchNormalization()(spatial)

    spatial = Local_Seq_Conv(output_dim=conv3_depth, seq_len=seq_len,
                             kernel_size=(conv_filter_size, conv_filter_size, conv2_depth, conv3_depth),
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    spatial = Activation(act_conv)(spatial)
    if batchnorm_on == 1:
        spatial = BatchNormalization()(spatial)

    if n_conv_layers > 3:
        spatial = Local_Seq_Conv(output_dim=conv4_depth, seq_len=seq_len,
                                 kernel_size=(conv_filter_size, conv_filter_size, conv3_depth, conv4_depth),
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                                 strides=(1, 1))(spatial)
        spatial = Activation(act_conv)(spatial)
        if batchnorm_on == 1:
            spatial = BatchNormalization()(spatial)

        if n_conv_layers > 4:
            spatial = Local_Seq_Conv(output_dim=conv5_depth, seq_len=seq_len,
                                     kernel_size=(conv_filter_size, conv_filter_size, conv4_depth, conv5_depth),
                                     kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                                     strides=(1, 1))(spatial)
            if pooling_on == 1:
                spatial = MaxPooling3D(pool_size=(1, pooling_size, pooling_size))(spatial)
            spatial = Activation(act_conv)(spatial)
            if batchnorm_on == 1:
                spatial = BatchNormalization()(spatial)

    spatial = Flatten()(spatial)
    spatial = Reshape(target_shape=(seq_len, -1))(spatial)
    spatial_out = Dense(units=conv_fc_units)(spatial)

    if n_lstm_layers == 0:
        spatial_out = Flatten()(spatial_out)
        lstm = Dense(units=lstm_units)(spatial_out)
        lstm = Activation('relu')(lstm)
    elif n_lstm_layers == 1:
        lstm = LSTM(units=lstm_units, batch_input_shape=(batch_size, seq_len, lstm_units), return_sequences=False)(
            spatial_out)
        lstm = Activation(act_lstm)(lstm)
    else:
        lstm = LSTM(units=lstm_units, batch_input_shape=(batch_size, seq_len, lstm_units), return_sequences=True)(
            spatial_out)
        lstm = Activation(act_lstm)(lstm)
        lstm = LSTM(units=lstm_units, return_sequences=False)(lstm)
        lstm = Activation(act_lstm)(lstm)

    if dropout_on == 1:
        res = Dropout(0.2)(lstm)
    else:
        res = lstm
    res = Dense(units=lstm_fc_units)(res)

    if n_fc_layers == 2:
        res = Dense(units=last_fc_units)(res)

    return res


def repeated_lstm(input, n_lstm_layers, seq_len, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units):
    lstm = Reshape(target_shape=(seq_len, -1))(input)

    if n_lstm_layers == 1:
        lstm = LSTM(units=lstm_units, batch_input_shape=(batch_size, seq_len, lstm_units), return_sequences=False)(lstm)
        lstm = Activation(act_lstm)(lstm)
    else:
        lstm = LSTM(units=lstm_units, batch_input_shape=(batch_size, seq_len, lstm_units), return_sequences=True)(lstm)
        lstm = Activation(act_lstm)(lstm)
        lstm = LSTM(units=lstm_units, return_sequences=False)(lstm)
        lstm = Activation(act_lstm)(lstm)

    if dropout_on == 1:
        res = Dropout(0.2)(lstm)
    else:
        res = lstm
    res = Dense(units=lstm_fc_units)(res)

    if n_fc_layers == 2:
        res = Dense(units=last_fc_units)(res)

    return res