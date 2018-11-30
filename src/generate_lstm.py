from dataIndices import *
import pandas as pd

def loadDataset(n_data, spdDict, times, links, forecasting_horizon, seq_len, temp_type='default'):
    spdDict_raw = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()

    result_x = []
    result_y = []
    result_y_raw = []
    for i in range(n_data):
        targetLink = links[i]
        targetTime = times[i]

        trange = dataTimeRange(targetTime, forecasting_horizon, seq_len, temp_type)
        truth = spdDict[targetLink][targetTime]
        result_x.append(spdDict[targetLink][trange])
        result_y.append(truth)
        result_y_raw.append(spdDict_raw[targetLink][targetTime])

    return (np.array(result_x), np.array(result_y), np.array(result_y_raw))

def generateTimeSeriesSet(data_type, forecasting_horizon, seq_len, spdDict,
                          n_train=50000, n_val=10000, n_test=10000, seed=821, temp_type='default'):

    if data_type < 20:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_hw.npy')
    elif data_type < 30:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx1.npy')
    else:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx2.npy')

    st = tt.time()
    print "######################################################################################################"
    print "Start to load data. (data_type: %i, forecasting_horizon: %i, seq_len: %i, temp_type: %s)" %(data_type, forecasting_horizon, seq_len, temp_type)
    (train_time, val_time, test_time, train_links, val_links, test_links) = datasetIdx(seed, links, n_train, n_val, n_test)
    (train_x, train_y, train_y_raw) = loadDataset(n_train, spdDict, train_time, train_links, forecasting_horizon, seq_len, temp_type)
    (val_x, val_y, val_y_raw) = loadDataset(n_val, spdDict, val_time, val_links, forecasting_horizon, seq_len, temp_type)
    (test_x, test_y, test_y_raw) = loadDataset(n_test, spdDict, test_time, test_links, forecasting_horizon, seq_len, temp_type)
    print train_x.shape
    print "Data loading is completed. (time: %i secs)" %(tt.time() - st)
    print "######################################################################################################"

    return (train_x, train_y, train_y_raw, val_x, val_y, val_y_raw, test_x, test_y, test_y_raw)

def findSpdLimits(links, spdLimitArray):
    result = []
    for l in links:
        result.append(spdLimitArray[spdLimitArray['LINK_ID'] == l]['highLimit'].values)
    return result


def generateSpdLimits(data_type, normalization_opt, n_train=50000, n_val=10000, n_test=10000, seed=821):
    if data_type < 20:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_hw.npy')
    elif data_type < 30:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx1.npy')
    else:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx2.npy')

    (train_time, val_time, test_time, train_links, val_links, test_links) = datasetIdx(seed, links, n_train, n_val, n_test)

    if normalization_opt == 0:
        if data_type >= 30:
            trainLimits = [236.01] * n_train
            valLimits = [236.01] * n_val
            testLimits = [236.01] * n_test
        elif data_type >= 20:
            trainLimits = [143.0] * n_train
            valLimits = [143.0] * n_val
            testLimits = [143.0] * n_test
    elif normalization_opt > 0:
        spdLimit = pd.read_csv('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/rawdata/link_information.csv')

        trainLimits = findSpdLimits(train_links, spdLimit)
        valLimits = findSpdLimits(val_links, spdLimit)
        testLimits = findSpdLimits(test_links, spdLimit)
    else:
        trainLimits = None
        valLimits = None
        testLimits = None

    return (trainLimits, valLimits, testLimits)

