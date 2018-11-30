from dataIndices import *

def mappingMatrix(data_type, img_size, targetLayer, traj_opt = 'all', excludeSelf = False):
    trajDict = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/' + str(traj_opt) + 'Traj.npy').item()

    if data_type >= 30:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx2.npy')
    elif data_type >= 20:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx1.npy')
    else:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_hw.npy')

    coords = datasetCoord(links, trajDict)
    if data_type == 11:
        layerIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/localIds/localIds_' + str(traj_opt) + '_hw_R' + str(int((img_size-1)/2)) + '.npy').item()
    elif data_type == 21:
        layerIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/localIds/localIds_' + str(traj_opt) + '_cx1_R' + str(int((img_size-1)/2)) + '.npy').item()
    elif data_type == 31:
        layerIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/localIds/localIds_' + str(traj_opt) + '_cx2_R' + str(int((img_size - 1) / 2)) + '.npy').item()
    elif data_type % 10 == 2:
        layerIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/layerIds_' + str(traj_opt) + '/' + str(int(data_type / 10)) + '3/layerIds_img' + str(img_size) + '.npy').item()
    else:
        layerIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/layerIds_' + str(traj_opt) + '/' + str(data_type) + '/layerIds_img' + str(img_size) + '.npy').item()

    result = {}
    target = 0
    layerIds_rev = {}
    for targetId in links:
        linkCoord = coords[target]

        if data_type % 10 == 2:
            neighbors = np.sort(layerIds[targetId][0])
        elif data_type % 10 == 1:
            neighbors = np.sort(layerIds[targetId])
        else:
            neighbors = np.sort(layerIds[targetId][targetLayer])

        if excludeSelf:
            idx = np.where(neighbors == targetId)
            neighbors = np.delete(neighbors, idx)

        neighbors_coords = []
        neighbors_ids = []

        k = 0
        for l in neighbors:
            neighbors_coords += trajDict[l].tolist()
            neighbors_ids += [k] * len(trajDict[l].tolist())
            k += 1

        freqMat = np.zeros([img_size, img_size], dtype=float)
        i = 0
        layerIds_ = []
        passingIds = {}
        for c in neighbors_coords:
            r = (img_size - 1) / 2
            proj_x = r - linkCoord[0]
            proj_y = r - linkCoord[1]
            linkId = neighbors_ids[i]

            if (c[0] + proj_x >= 0) & (c[0] + proj_x < img_size) & (c[1] + proj_y >= 0) & (c[1] + proj_y < img_size):
                freqMat[c[0] + proj_x, c[1] + proj_y] += 1
                layerIds_.append(neighbors[linkId])
                if linkId in passingIds.keys():
                    passingIds[linkId] += [(c[0]+proj_x, c[1]+proj_y)]
                else:
                    passingIds.update({linkId: [(c[0]+proj_x, c[1]+proj_y)]})

            i += 1

        layerIds_rev.update({targetId: np.unique(layerIds_)})

        rev = np.reciprocal(freqMat)
        rev[rev == np.inf] = 0

        normMat = []
        for j in passingIds.keys():
            vec = [0] * img_size*img_size
            for c in passingIds[j]:
                vec[img_size * c[0] + c[1]] += rev[c]
            normMat.append(vec)

            result.update({targetId: np.array(normMat)})

        target += 1

    return (result, layerIds_rev)


def loadData(n_data, times, links, layerIds, spdIds, spdArray, normalizeMatrix, forecasting_horizon, seq_len, img_size, is_image = True, temp_type='default'):
    result_x = []
    # count = 0.0

    for l in range(n_data):
        targetLink = links[l]
        targetTime = times[l]
        targetLayerIds = np.sort(layerIds[targetLink])
        # count += len(targetLayerIds)

        trange = dataTimeRange(targetTime, forecasting_horizon, seq_len, temp_type)
        result = []
        for i in targetLayerIds:
            linkMatrix = []
            for t in trange:
                linkMatrix.append(spdArray[np.where(i == spdIds)[0][0], t])
            result.append(linkMatrix)

        if targetLink in normalizeMatrix.keys():
            result_x.append(np.matmul(np.transpose(normalizeMatrix[targetLink]), np.array(result)))
        else:
            if is_image:
                result_x.append(np.zeros([img_size*img_size, seq_len]))
            else:
                result_x.append(np.zeros([1, seq_len]).tolist()[0])

    # print float(count / n_data)
    if is_image:
        return np.array(result_x).reshape([n_data, img_size, img_size, seq_len])
    else:
        return np.array(result_x)

def loadLabels(n_data, spdIds, spdArray, spdDict_raw, times, links):
    result_y = []
    result_y_raw = []
    for l in range(n_data):
        targetLink = links[l]
        targetTime = times[l]
        targetRow = np.where(spdIds == targetLink)[0][0]

        truth = spdArray[targetRow, targetTime]

        result_y.append(truth)
        result_y_raw.append(spdDict_raw[targetLink][targetTime])

    return (np.array(result_y), np.array(result_y_raw))

def loadSpdIdsAndRawDictAndLinks(normalization_opt, data_type):
    if normalization_opt == 'raw':
        spdIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/all_links_afterFiltering2.npy')
    else:
        spdIds = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spdIds_spdLimitNorm_' + str(normalization_opt) + '.npy')

    spdDict_raw = np.load(
        '/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()

    if data_type < 20:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_hw.npy')
    elif data_type < 30:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx1.npy')
    else:
        links = np.load('/home/keun/PycharmProjects/trafficPrediction/src/preprocessing/data/linkIds_cx2.npy')

    return (spdIds, spdDict_raw, links)


def generateImageset(data_type, img_size, forecasting_horizon, seq_len, spdArray,
                     n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt='all', normalization_opt='raw', temp_type=None):

    print "######################################################################################################"
    print("Start to load data. (data_type: %i, img_size: %i, forecasting_horizon: %i, seq_len: %i)" %(data_type, img_size, forecasting_horizon, seq_len))
    loading_st = tt.time()

    (spdIds, spdDict_raw, links) = loadSpdIdsAndRawDictAndLinks(normalization_opt, data_type)

    if data_type % 10 < 3:
        n_layers = 1
    else:
        n_layers = 4

    (train_time, val_time, test_time, train_links, val_links, test_links) = datasetIdx(seed, links, n_train, n_val, n_test)
    train_x = []
    val_x = []
    test_x = []
    for layer in range(n_layers):
        st = tt.time()
        (normalizeMatrix, layerIds) = mappingMatrix(data_type, img_size, layer, traj_opt)
        train_x_part = loadData(n_train, train_time, train_links, layerIds, spdIds, spdArray, normalizeMatrix, forecasting_horizon, seq_len, img_size, True, temp_type)
        val_x_part = loadData(n_val, val_time, val_links, layerIds, spdIds, spdArray, normalizeMatrix, forecasting_horizon, seq_len, img_size, True, temp_type)
        test_x_part = loadData(n_test, test_time, test_links, layerIds, spdIds, spdArray, normalizeMatrix, forecasting_horizon, seq_len, img_size, True, temp_type)

        train_x.append(train_x_part)
        val_x.append(val_x_part)
        test_x.append(test_x_part)

        print(("Layer %i is completed. (time: %i secs)" %(layer, tt.time() - st)))

    train_x = np.array(train_x)
    train_x = np.moveaxis(train_x, 0, 4)
    train_x = np.moveaxis(train_x, 3, 1)

    val_x = np.array(val_x)
    val_x = np.moveaxis(val_x, 0, 4)
    val_x = np.moveaxis(val_x, 3, 1)

    test_x = np.array(test_x)
    test_x = np.moveaxis(test_x, 0, 4)
    test_x = np.moveaxis(test_x, 3, 1)

    (train_y, train_y_raw) = loadLabels(n_train, spdIds, spdArray, spdDict_raw, train_time, train_links)
    (val_y, val_y_raw) = loadLabels(n_val, spdIds, spdArray, spdDict_raw, val_time, val_links)
    (test_y, test_y_raw) = loadLabels(n_test, spdIds, spdArray, spdDict_raw, test_time, test_links)

    print train_x.shape
    print("End to load data. (time: %i secs)" %(tt.time() - loading_st))
    print "######################################################################################################"

    return (train_x, np.array(train_y), np.array(train_y_raw), val_x, np.array(val_y), np.array(val_y_raw), test_x, np.array(test_y), np.array(test_y_raw))


def loadTargetData(n_data, links, times, forecasting_horizon, seq_len, spdDict, temp_type=None):
    result = []
    for i in range(n_data):
        targetLink = links[i]
        targetTime = times[i]

        trange = dataTimeRange(targetTime, forecasting_horizon, seq_len, temp_type)
        result.append(spdDict[targetLink][trange])
    return np.array(result).reshape([n_data, seq_len, 1])

def generateNdVector(data_type, img_size, forecasting_horizon, seq_len, spdArray, spdDict,
                     n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt='all', normalization_opt='raw', temp_type=None):

    print "######################################################################################################"
    print("Start to load data. (data_type: %i, img_size: %i, forecasting_horizon: %i, seq_len: %i)" %(data_type, img_size, forecasting_horizon, seq_len))
    loading_st = tt.time()

    (spdIds, spdDict_raw, links) = loadSpdIdsAndRawDictAndLinks(normalization_opt, data_type)

    if data_type % 10 < 3:
        n_layers = 1
    else:
        n_layers = 4

    (train_time, val_time, test_time, train_links, val_links, test_links) = datasetIdx(seed, links, n_train, n_val, n_test)
    train_x = loadTargetData(n_train, train_links, train_time, forecasting_horizon, seq_len, spdDict, temp_type)
    val_x = loadTargetData(n_val, val_links, val_time, forecasting_horizon, seq_len, spdDict, temp_type)
    test_x = loadTargetData(n_test, test_links, test_time, forecasting_horizon, seq_len, spdDict, temp_type)

    for layer in range(n_layers):
        st = tt.time()
        (normalizeMatrix, layerIds) = mappingMatrix(data_type, img_size, layer, traj_opt=traj_opt, excludeSelf=True)
        normalizeVector = {}
        for l in layerIds.keys():
            if len(layerIds[l]) > 0:
                normalizeVector.update({l: [1 / len(layerIds[l])] * len(layerIds[l])})

        train_x_part = loadData(n_train, train_time, train_links, layerIds, spdIds, spdArray, normalizeVector, forecasting_horizon, seq_len, img_size, False, temp_type).reshape([n_train, seq_len, 1])
        val_x_part = loadData(n_val, val_time, val_links, layerIds, spdIds, spdArray, normalizeVector, forecasting_horizon, seq_len, img_size, False, temp_type).reshape([n_val, seq_len, 1])
        test_x_part = loadData(n_test, test_time, test_links, layerIds, spdIds, spdArray, normalizeVector, forecasting_horizon, seq_len, img_size, False, temp_type).reshape([n_test, seq_len, 1])
        train_x = np.dstack([train_x, train_x_part])
        val_x = np.dstack([val_x, val_x_part])
        test_x = np.dstack([test_x, test_x_part])

        print(("Layer %i is completed. (time: %i secs)" %(layer, tt.time() - st)))

    train_x = np.array(train_x)
    val_x = np.array(val_x)
    test_x = np.array(test_x)

    (train_y, train_y_raw) = loadLabels(n_train, spdIds, spdArray, spdDict_raw, train_time, train_links)
    (val_y, val_y_raw) = loadLabels(n_val, spdIds, spdArray, spdDict_raw, val_time, val_links)
    (test_y, test_y_raw) = loadLabels(n_test, spdIds, spdArray, spdDict_raw, test_time, test_links)

    print train_x.shape
    print("End to load data. (time: %i secs)" %(tt.time() - loading_st))
    print "######################################################################################################"

    return (train_x, np.array(train_y), np.array(train_y_raw), val_x, np.array(val_y), np.array(val_y_raw), test_x, np.array(test_y), np.array(test_y_raw))


'''
# sample code

data_type = 25
img_size = 21
forecasting_horizon = 1
seq_len = 1
spdIds = np.load('preprocessing/data/all_links_afterFiltering2.npy')
allTraj = np.load('preprocessing/data/allTraj.npy').item()
spdArray = np.load('preprocessing/data/spdArray.npy')
spdDict = np.load('preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()
spd_raw = np.load('preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()

(train_x, train_y, _, val_x, val_y, _, test_x, test_y, _) = generateImageset(data_type, img_size, forecasting_horizon, seq_len, allTraj, spdIds, spdArray, spd_raw,
                                                                             n_train=50000, n_val=10000, n_test=10000, seed=821, traj_opt='all', temp_type='default')
# (train_x, train_y, _, val_x, val_y, _, test_x, test_y, _) = generateNdVector(data_type, img_size, forecasting_horizon, seq_len, allTraj, spdIds, spd, spdDict, spd_raw,
#                                                                              n_train=50000, n_val=10000, n_test=10000, seed=821, temp_type='default')

import matplotlib.pyplot as plt

# plt.imshow(train_x[0, 0, :, :, 0])
# plt.show()


for layer in range(4):
    plt.imshow(train_x[0, 0, :, :, layer])
    plt.show()
'''