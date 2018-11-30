#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd

year = 2016

if year == 2018:
    month = range(1, 11)
else:
    month = range(1, 13)

data = pd.DataFrame()
for m in month:
    if m < 10:
        m = str(0) + str(m)
    f = str(year) + str(m) + '.csv'
    print f
    try:
        newdata = pd.read_csv('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/rawdata/speed/' + f, encoding='UTF8', error_bad_lines=False, index_col=False)
    except UnicodeDecodeError:
        newdata = pd.read_csv('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/rawdata/speed/' + f, encoding='CP949', error_bad_lines=False, index_col=False)

    data = pd.concat([data, newdata])
    print data.shape
    print data.tail(1)

# print data.columns
if year == 2018:
    # year: 2018 (n_col=36)
    data.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 'length',
                    'none1', 'none2', 'road', 'none3', 'LINK_ID', 'direction', 'start', 'weekday', 'date', 'end',
                    'none4']
else:
    # year: 2014, 2015, 2016, 2017 (n_col=32)
    data.columns = ['date', 'weekday', 'road', 'LINK_ID', 'start', 'end', 'length', 'direction', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

data = data[['date', 'LINK_ID', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]].sort_values(['date', 'LINK_ID']).reset_index(drop=True)

days = len(np.unique(data['date']))
links = np.unique(data['LINK_ID'].values)
spd = {}
for l in links:
    ldata = data[data['LINK_ID'] == l].sort_values('date').reset_index(drop=True)[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]].values
    if ldata.shape[0] == days:
        try:
            spd.update({l: ldata.flatten().astype('float64')})
        except UnicodeEncodeError:
            print l
            print ldata.flatten()
            #1240028500

# print len(links)
# print len(spd)
np.save('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_' + str(year) + '.npy', spd)