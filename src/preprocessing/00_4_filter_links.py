import numpy as np

'''
link filtering based on speed data
- criteria 1: More than 6 consecutive 0s.
- criteria 2: More than 50% 0s per year.
'''

links = np.load('data/all_links.npy')

del_links1 = []
del_links2 = []
spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_augmented_minusWhenAllZero.npy').item()

del_links = []
result = []
for l in links:
    dt = np.nan_to_num(spd[l]).astype('float')
    zero_rows = np.where(dt == 0)[0]

    criteria_1 = 0.0
    for row in zero_rows:
        if row < (dt.shape[0]-5):
            if ((dt[row] == 0) & (dt[row+1] == 0) & (dt[row+2] == 0) & (dt[row+3] == 0) & (dt[row+4] == 0) & (dt[row+5] == 0)):
                criteria_1 += 1.0
                result += [row]

    criteria_2 = sum(dt == 0)

    if criteria_1 > 0:
        del_links += [l]
    elif criteria_2 > (0.5 * len(dt)):
        del_links += [l]

print len(np.unique(result))
print np.unique(result)
del_links = np.unique(del_links)
print len(del_links)

links = [l for l in links if l not in del_links]
print len(links)

np.save('data/all_links_afterFiltering.npy', links)