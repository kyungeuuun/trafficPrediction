import numpy as np
import pandas as pd
from scipy import stats

spdDict = np.load('data/spd_interpolated_ignoreErrTimes.npy').item()
spdArray = np.load('data/spdArray.npy')
spdIds = np.load('data/all_links_afterFiltering2.npy').tolist()

spdLimit = pd.read_csv('rawdata/link_information.csv')

cx1_links = np.load('data/linkIds_cx1.npy')
cx2_links = np.load('data/linkIds_cx2.npy')

# maxList = []
# for l in cx1_links:
#     maxList.append(max(spdArray[np.where(l == spdIds)[0][0], :35064]))
# print max(maxList) #143
#
# maxList = []
# for l in cx2_links:
#     maxList.append(max(spdArray[np.where(l == spdIds)[0][0], :35064]))
# print max(maxList) #236.01
#
# spdLimit_noInfoIds = spdLimit[spdLimit['highLimit'] == 0]['LINK_ID'].values
#
# for l in cx1_links:
#     if l in spdLimit_noInfoIds:
#         print l #no
#
# for l in cx2_links:
#     if l in spdLimit_noInfoIds:
#         print l  # no
#
# #ok!
#
#
# print np.max(spdArray) #312
# print spdArray.shape #3790, 42360

for clipped in [True, False]:
    for dataType in [2, 3]:
        for limitCoef in [1, 2]:
            spdArray_limitNormed = []
            spdIds_rearranged = []

            if dataType == 2:
                links = cx1_links
            else:
                links = cx2_links

            j = 0
            result = []
            for l in links:
                i = np.where(l == spdIds)[0]
                limit = spdLimit[spdLimit['LINK_ID'] == l]['highLimit'].values[0] * limitCoef

                normed = spdArray[i, :] / limit
                if clipped:
                    spdArray_limitNormed.append(normed.clip(max=1))
                else:
                    spdArray_limitNormed.append(normed)

                spdIds_rearranged.append(l)

            #     count = sum(normed > 1)
            #     if count > 0:
            #         result.append(count)
            #         j += 1
            #

            if clipped:
                spdArray_limitNormed = np.array(spdArray_limitNormed).squeeze()
                print spdArray_limitNormed.shape
                np.save('data/spdArray_spdLimitNorm_' + str(limitCoef) + '_clipped_' + str(dataType) + '.npy', spdArray_limitNormed)
                np.save('data/spdIds_spdLimitNorm_' + str(limitCoef) + '_clipped_' + str(dataType) + '.npy', spdIds_rearranged)
            else:
                spdArray_limitNormed = np.array(spdArray_limitNormed).squeeze()
                print spdArray_limitNormed.shape
                np.save('data/spdArray_spdLimitNorm_' + str(limitCoef) + '_' + str(dataType) + '.npy', spdArray_limitNormed)
                np.save('data/spdIds_spdLimitNorm_' + str(limitCoef) + '_' + str(dataType) + '.npy', spdIds_rearranged)

            # import matplotlib.pyplot as plt
            # plt.hist(result)
            # plt.show()
            # print '####################'
            # print limitCoef
            # print j
            # if len(result) > 0:
            #     print 'min: %i, median: %i, mode: %i, mean: %i, max: %i' %(min(result), np.median(result), stats.mode(result, axis=None)[0][0], np.mean(result), max(result))

        for limitCoef in [1, 2]:
            spdDict_limitNormed = {}
            spdDict_raw_limitNormed = {}

            j = 0
            result = []
            for l in links:
                i = np.where(l == spdIds)[0]
                limit = spdLimit[spdLimit['LINK_ID'] == l]['highLimit'].values[0] * limitCoef

                normed = spdArray[i, :] / limit
                if clipped:
                    spdDict_limitNormed.update({l: normed.clip(max=1).squeeze()})
                else:
                    spdDict_limitNormed.update({l: normed.squeeze()})
                spdDict_raw_limitNormed.update({l: spdArray[i, :].squeeze()})

            if clipped:
                np.save('data/spdDict_spdLimitNorm_' + str(limitCoef) + '_clipped_' + str(dataType) + '.npy', spdDict_limitNormed)
            else:
                np.save('data/spdDict_spdLimitNorm_' + str(limitCoef) + '_' + str(dataType) + '.npy', spdDict_limitNormed)

            np.save('data/spdDict_raw_spdLimitNorm_' + str(limitCoef) + '_' + str(dataType) + '.npy', spdDict_raw_limitNormed)


clipped = False
for dataType in [2, 3]:
    spdArray_limitNormed = []
    spdIds_rearranged = []

    if dataType == 2:
        links = cx1_links
    else:
        links = cx2_links

    # j = 0
    # result = []
    for l in links:
        i = np.where(l == spdIds)[0]

        if dataType == 2:
            limit = 143
        else:
            limit = 236.01

        normed = spdArray[i, :] / limit
        if clipped:
            spdArray_limitNormed.append(normed.clip(max=1))
        else:
            spdArray_limitNormed.append(normed)

        spdIds_rearranged.append(l)

    #     count = sum(normed > 1)
    #     if count > 0:
    #         result.append(count)
    #         j += 1
    #

    spdArray_limitNormed = np.array(spdArray_limitNormed).squeeze()
    print spdArray_limitNormed.shape

    np.save('data/spdArray_spdLimitNorm_0_' + str(dataType) + '.npy', np.array(spdArray_limitNormed))
    np.save('data/spdIds_spdLimitNorm_0_' + str(dataType) + '.npy', spdIds_rearranged)

for dataType in [2, 3]:
    spdDict_limitNormed = {}
    spdDict_raw_limitNormed = {}

    if dataType == 2:
        links = cx1_links
    else:
        links = cx2_links

    j = 0
    result = []
    for l in links:
        i = np.where(l == spdIds)[0]
        if dataType == 2:
            limit = 143
        else:
            limit = 236.01

        normed = spdArray[i, :] / limit
        if clipped:
            spdDict_limitNormed.update({l: normed.clip(max=1).squeeze()})
        else:
            spdDict_limitNormed.update({l: normed.squeeze()})
        spdDict_raw_limitNormed.update({l: spdArray[i, :].squeeze()})

    np.save('data/spdDict_spdLimitNorm_0_' + str(dataType) + '.npy', spdDict_limitNormed)
    np.save('data/spdDict_raw_spdLimitNorm_0_' + str(dataType) + '.npy', spdDict_raw_limitNormed)