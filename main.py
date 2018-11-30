import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '/home/keun/PycharmProjects/trafficPrediction/src/')
from basicModels import *
import os


'''
n_train=50000
n_val=10000
n_test=10000
seed=821
temp_type=None
'''

def main():
    num_trials = 1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sess = tf.Session()
    K.set_session(sess)

    models = pd.read_csv('/home/keun/PycharmProjects/trafficPrediction/model/' + str(EXP_) + '.csv')

    for index, row in models.iterrows():
        data_type = int(row['data_type'])
        img_size = int(row['img_size'])
        input_depth = int(row['img_depth'])
        forecasting_horizon = int(row['forecasting_horizon'])
        seq_len = int(row['seq_len'])
        model_type = str(row['model_type'])

        batch_size = int(row['batch_size'])
        max_epoch = int(row['max_epoch'])
        # max_epoch = 1
        optimizer = str(row['optimizer'])
        opt_learningrate = float(row['opt_learningrate'])
        batchnorm_on = int(row['batchnorm_on']) #0 or 1
        dropout_on = int(row['dropout_on']) #0 or 1
        loss_func = str(row['loss_func'])
        earlystop_on = int(row['earlystop_on']) #0 or 1

        n_conv_layers = int(row['n_conv_layers'])
        conv1_depth = int(row['conv1_depth'])
        conv2_depth = int(row['conv2_depth'])
        conv3_depth = int(row['conv3_depth'])
        conv4_depth = int(row['conv4_depth'])
        conv5_depth = int(row['conv5_depth'])
        conv_fc_units = int(row['conv_fc_units'])
        act_conv = str(row['act_conv'])
        pooling_on = int(row['pooling_on'])
        pooling_size = int(row['pooling_size'])
        conv_filter_size = int(row['conv_filter_size'])

        n_lstm_layers = int(row['n_lstm_layers'])
        lstm_units = int(row['lstm_units'])
        act_lstm = str(row['act_lstm'])
        lstm_fc_units = int(row['lstm_fc_units'])

        n_fc_layers = int(row['n_fc_layers'])
        last_fc_units = int(row['last_fc_units'])

        temp_type = str(row['temp_type'])
        traj_opt = str(row['traj_opt'])
        normalization_opt = str(row['normalization_opt'])

        if normalization_opt == 'raw':
            spdDict = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()
            spdArray = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spdArray.npy')
        else:
            spdDict = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spdDict_spdLimitNorm_' + str(normalization_opt) + '.npy').item()
            spdArray = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spdArray_spdLimitNorm_' + str(normalization_opt) + '.npy')

        param_setup = pd.DataFrame(columns=param_colnames())
        param_setup.loc[index] = [forecasting_horizon, data_type, seq_len, img_size, batch_size, max_epoch, optimizer, opt_learningrate, batchnorm_on, dropout_on, loss_func, earlystop_on,
                                  n_conv_layers, conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                                  n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, temp_type, traj_opt, normalization_opt]

        for trial_no in range(num_trials):
            EXP_NO = EXP_ + '/' + str(forecasting_horizon) + 'hours_dataType' + str(data_type) + '_normalization' + str(normalization_opt) + '_traj' + str(traj_opt) + \
                     '_img' + str(img_size) + '_seq' + str(seq_len) + '_trial' + str(trial_no)

            print('-----------------------------------------------------------------------------')
            print(EXP_NO)
            print(str(float(index) + 1) + ' / ' + str(float(models.shape[0])) + ' / trial : ' + str(trial_no))
            print('data_type: %i, seq_len: %i, img_size: %i, normalization: %s, traj: %s' %(data_type, seq_len, img_size, normalization_opt, traj_opt))
            print('-----------------------------------------------------------------------------')

            if model_type == 'lstm':
                lstm_model(EXP_NO, data_type, forecasting_horizon, seq_len, spdDict,
                           n_lstm_layers, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units,
                           opt_learningrate, loss_func, max_epoch,
                           n_train=100000, n_val=10000, n_test=10000, seed=821, temp_type=temp_type, normalization_opt=normalization_opt)

            elif model_type == 'cnnlstm':
                cnnlstm_model(EXP_NO, data_type, img_size, input_depth, forecasting_horizon, seq_len, spdArray,
                              batch_size, batchnorm_on, dropout_on, n_conv_layers, conv1_depth, conv2_depth,
                              conv3_depth, conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                              n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                              n_train=100000, n_val=10000, n_test=10000, seed=821, traj_opt=traj_opt, temp_type=temp_type, normalization_opt=normalization_opt)

            elif model_type == 'ndlstm':
                nd_lstm_model(EXP_NO, data_type, img_size, spdDict, forecasting_horizon, seq_len, spdArray, batch_size, dropout_on,
                              n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                              n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt=traj_opt, temp_type=temp_type, normalization_opt=normalization_opt)

            elif model_type == '3dcnn':
                cnn3d_model(EXP_NO, data_type, img_size, forecasting_horizon, seq_len, spdArray, batch_size,
                            batchnorm_on, dropout_on, n_conv_layers, conv1_depth, conv2_depth, conv3_depth,
                            conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                            lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                            n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt=traj_opt, temp_type=temp_type, normalization_opt=normalization_opt)


EXP_ = 'test_datasize'
gpu_id = 1
if not os.path.exists('/home/keun/PycharmProjects/trafficPrediction/log/' + EXP_):
    os.makedirs('/home/keun/PycharmProjects/trafficPrediction/log/' + EXP_)
if __name__ == '__main__':
    main()
