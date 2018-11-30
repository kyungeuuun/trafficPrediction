import numpy as np
import os
import time as tt

def dataTimeRange(pred_time, forecasting_horizon, seq_len, temp_type='default'):
    # if temp_type == 'day':
    #     if ((pred_time - 288) % 2016 < 288) & ((pred_time - 288) % 2016 > 1727):
    #         return range(pred_time - 288*3 - (int(seq_len * 0.5) - 1), pred_time - 288*3 + int(seq_len * 0.5) + 1)
    #     else:
    #         return range(pred_time - 288 - (int(seq_len * 0.5) - 1), pred_time - 288 + int(seq_len * 0.5) + 1)
    #
    # elif temp_type == 'week':
    #     return range(pred_time - 2016 - (int(seq_len * 0.5) - 1), pred_time - 2016 + int(seq_len * 0.5) + 1)
    #
    # else:
    return range(pred_time - forecasting_horizon + 1 - seq_len, pred_time - forecasting_horizon + 1)

def datasetIdx(seed, links, n_train, n_val, n_test):
    np.random.seed(seed)

    # data path problem...........
    train_indices = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/train_dateIdx.npy')
    test_indices = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/test_dateIdx.npy')

    train_time = np.random.choice(train_indices, n_train, replace=True)
    val_time = np.random.choice(train_indices, n_val, replace=True)
    test_time = np.random.choice(test_indices, n_test, replace=True) + 35064

    train_links = np.array(links)[np.random.choice(len(links), n_train, replace=True)]
    val_links = np.array(links)[np.random.choice(len(links), n_val, replace=True)]
    test_links = np.array(links)[np.random.choice(len(links), n_test, replace=True)]

    return (train_time, val_time, test_time, train_links, val_links, test_links)

def selectMiddleCoord(linkId, coord):
    if len(coord[linkId].shape) > 1:
        return coord[linkId][int(len(coord[linkId]) * 0.5)]
    else:
        return coord[linkId]

def datasetCoord(links, coord):
    result = []
    for l in links:
        result.append(selectMiddleCoord(l, coord))
    return np.array(result)