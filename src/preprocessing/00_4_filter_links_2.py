import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_augmented_minusWhenAllZero.npy').item()
links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/all_links_afterFiltering.npy')
print len(links)

# del_links = []
# err = []
# for l in links:
#     lspd = spd[l].reshape([-1, 24])
#     uniq_rows = np.unique(lspd, axis=0)
#
#     if lspd.shape[0] > uniq_rows.shape[0]:
#         del_links += [l]
#
# print np.unique(del_links)
# print len(np.unique(del_links))
# # np.save('data/del_links_dailySpeed.npy', del_links)
# # np.save('data/all_links_afterFiltering2.npy', [l for l in links if l not in del_links])
#
#
# result = pd.DataFrame({'LINK_ID': [], 'row1': [], 'row2': []})
# k = 0
# st = time.time()
# for l in del_links:
#     dt = spd[l].reshape([-1, 24])
#     print dt.shape
#
#     for i in range(dt.shape[0]):
#         for j in range(i+1, dt.shape[0]):
#             if (sum(dt[i, :] - dt[j, :]) == 0) & (i != j):
#                 # print '###### row: %i and row: %i ######' %(i, j)
#                 result.loc[k] = [l, i, j]
#                 k += 1
#
#                 if k % 10000 == 0:
#                     print k
#                     print time.time() - st
#
# print result.head()
# result.to_csv('result.csv')

data = pd.read_csv('result.csv', index_col=0)
data['diff'] = data['row1'] - data['row2']
data['mod'] = data['diff'] % 7
# print np.unique(data['mod'].values)
print float(float(data[data['mod'] > 0].shape[0]) / float(data.shape[0]))
print data.groupby(['LINK_ID']).agg(['count'])