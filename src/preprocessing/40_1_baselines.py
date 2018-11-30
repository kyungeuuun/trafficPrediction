import numpy as np
import pandas as pd
from sklearn import linear_model
import sys
sys.path.insert(0, '../')
from dataIndices import *

def perfMetrics(err, truth):
    mae = np.mean(np.abs(err))
    mape = np.true_divide(np.abs(err), truth)
    mape = mape[~np.isnan(mape)]
    mape = np.mean(mape)
    rmse = np.sqrt(np.mean(np.square(err)))

    return (mape, mae, rmse)

# cx1_links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx1.npy')
cx2_links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx2.npy')
spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()

# (train_time, val_time, test_time, train_links, val_links, test_links) = datasetIdx(821, cx1_links, 50000, 10000, 10000)
(train_time, val_time, test_time, train_links, val_links, test_links) = datasetIdx(821, cx2_links, 50000, 10000, 10000)

#historical average
result = pd.DataFrame(columns=('baseline', 'mape', 'mae', 'rmse'))

print 'all'
err = []
truth = []
for l in range(10000):
    dt = spd[test_links[l]]
    ha_range = range(test_time[l] - 24*7, 0, -(24*7))
    err.append(np.mean(dt[ha_range]) - dt[test_time[l]])
    truth.append(dt[test_time[l]])

print perfMetrics(err, truth)
result.loc[0] = ['all', perfMetrics(err, truth)[0], perfMetrics(err, truth)[1], perfMetrics(err, truth)[2]]

print 'month'
err = []
truth = []
for l in range(10000):
    dt = spd[test_links[l]]
    ha_range = range(test_time[l] - 24*7, max(0, test_time[l]-24*7*4-1), -(24*7))
    err.append(np.mean(dt[ha_range]) - dt[test_time[l]])
    truth.append(dt[test_time[l]])
print perfMetrics(err, truth)
result.loc[1] = ['month', perfMetrics(err, truth)[0], perfMetrics(err, truth)[1], perfMetrics(err, truth)[2]]

print 'week'
err = []
truth = []
for l in range(10000):
    dt = spd[test_links[l]]
    err.append((dt[test_time[l] - 24*7]) - dt[test_time[l]])
    truth.append(dt[test_time[l]])
print perfMetrics(err, truth)
result.loc[2] = ['week', perfMetrics(err, truth)[0], perfMetrics(err, truth)[1], perfMetrics(err, truth)[2]]

print 'day'
err = []
truth = []
for l in range(10000):
    dt = spd[test_links[l]]
    err.append((dt[test_time[l] - 24]) - dt[test_time[l]])
    truth.append(dt[test_time[l]])
print perfMetrics(err, truth)
result.loc[3] = ['day', perfMetrics(err, truth)[0], perfMetrics(err, truth)[1], perfMetrics(err, truth)[2]]

print 'hour'
err = []
truth = []
for l in range(10000):
    dt = spd[test_links[l]]
    err.append((dt[test_time[l] - 1]) - dt[test_time[l]])
    truth.append(dt[test_time[l]])
print perfMetrics(err, truth)
result.loc[4] = ['hour', perfMetrics(err, truth)[0], perfMetrics(err, truth)[1], perfMetrics(err, truth)[2]]

print 'linear regression'

def baseline_LR(spd, train_links, train_times, test_links, test_times, seq_len, forecasting_horizon):
    x = []
    y = []
    for l in range(50000):
        trange = dataTimeRange(train_times[l], forecasting_horizon, seq_len)
        x.append(spd[train_links[l]][trange])
        y.append(spd[train_links[l]][train_times[l]])
    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    x = []
    y = []
    for l in range(len(test_links)):
        trange = dataTimeRange(test_times[l], forecasting_horizon, seq_len)

        x.append(spd[test_links[l]][trange])
        y.append(spd[test_links[l]][test_times[l]])
    pred = reg.predict(x)
    err = np.array(y) - np.array(pred)
    (mape, mae, rmse) = perfMetrics(err, np.array(y))

    print("LR performance : MAPE %.3f / MAE %.3f / RMSE %.3f" % (mape, mae, rmse))

    return (mape, mae, rmse)

print baseline_LR(spd, train_links, train_time, test_links, test_time, 6, 1)
lr = baseline_LR(spd, train_links, train_time, test_links, test_time, 6, 1)
result.loc[5] = ['lr', lr[0], lr[1], lr[2]]

print result

# result.to_csv('data/40_1_baseline_result.csv')
# print result