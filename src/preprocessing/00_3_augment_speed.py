import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_2014.npy').item().keys()
# print len(links)
# for year in [2015, 2016, 2017, 2018]:
#     spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_' + str(year) + '.npy').item()
#     links = list(set(links) & set(spd.keys()))
#     print len(links)
#
# np.save('data/all_links.npy', links)

links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/all_links.npy')

augSpd = {}
for year in [2014, 2015, 2016, 2017, 2018]:
    print year
    spd = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_' + str(year) + '_minusWhenAllZero.npy').item()

    # print len(spd[spd.keys()[0]]) #8760, 8760, 8784(2.29), 8760, 7296
    for l in links:
        newdata = np.nan_to_num(spd[l])

        if l in augSpd.keys():
            augSpd[l] = np.concatenate([augSpd[l], newdata])
        else:
            augSpd.update({l: np.nan_to_num(spd[l])})


# np.save('data/spd_augmented.npy', augSpd)
np.save('data/spd_augmented_minusWhenAllZero.npy', augSpd)