import numpy as np
import pandas as pd

links = np.load('data/all_links.npy')
# print len(links)

def checkWhenAllZeros(speedDict, links):
    data = []
    for l in links:
        data.append(np.nan_to_num(speedDict[l]))
    check = np.mean(np.array(data), axis=0)
    return np.where(check == 0)[0]

# spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_2015.npy').item()
# spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_2016.npy').item()

for year in [2014, 2015, 2016, 2017, 2018]:
    print '######'
    print year
    spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_' + str(year) + '.npy').item()
    idx = checkWhenAllZeros(spd, links)
    print len(idx)

    times = pd.date_range(start='1/1/' + str(year), end='1/1/' + str(year+1), closed='left', freq='1H')
    print times[idx]

    consecutive_idx = []
    for i in idx:
        if ((i-1) in idx) | ((i+1) in idx):
            consecutive_idx += [i]

    for l in links:
        spd[l][consecutive_idx] = -1

    np.save('data/spd_' + str(year) + '_minusWhenAllZero.npy', spd)



# times = pd.date_range(start='1/1/2015', end='1/1/2016', closed='left', freq='1H')
# print times[4246:4252]
# DatetimeIndex(['2015-06-26 22:00:00', '2015-06-26 23:00:00', '2015-06-27 00:00:00', '2015-06-27 01:00:00',
#                '2015-06-27 02:00:00', '2015-06-27 03:00:00'], dtype='datetime64[ns]', freq='H')

# times = pd.date_range(start='1/1/2016', end='1/1/2017', closed='left', freq='1H')
# print times[5062:5070]
# DatetimeIndex(['2016-07-29 22:00:00', '2016-07-29 23:00:00', '2016-07-30 00:00:00', '2016-07-30 01:00:00',
#                '2016-07-30 02:00:00', '2016-07-30 03:00:00', '2016-07-30 04:00:00', '2016-07-30 05:00:00'], dtype='datetime64[ns]', freq='H')

# times = pd.date_range(start='1/1/2017', end='1/1/2018', closed='left', freq='1H')
# print times[7583:7589]
# DatetimeIndex(['2017-11-12 23:00:00', '2017-11-13 00:00:00', '2017-11-13 01:00:00', '2017-11-13 02:00:00',
#                '2017-11-13 03:00:00', '2017-11-13 04:00:00'], dtype='datetime64[ns]', freq='H')
# print times[8380:8388]
# DatetimeIndex(['2017-12-16 04:00:00', '2017-12-16 05:00:00', '2017-12-16 06:00:00', '2017-12-16 07:00:00',
#                '2017-12-16 08:00:00', '2017-12-16 09:00:00', '2017-12-16 10:00:00', '2017-12-16 11:00:00'], dtype='datetime64[ns]', freq='H')



