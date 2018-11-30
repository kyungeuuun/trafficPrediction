import math
import numpy as np
import pandas as pd
import time as tt

pi = math.pi

'''
NEWS (13, 23, 33)
'''
def layerIds_NEWS(linkId, thetaDF, localId, theta_bd=15):
    bd1 = np.cos((theta_bd * pi / 180))
    bd2 = np.cos((90 - theta_bd) * pi / 180)

    baseTheta = thetaDF[thetaDF['LINK_ID'] == linkId]['theta'].values[0]
    localTheta = thetaDF[thetaDF['LINK_ID'].isin(localId)]
    localTheta['projTheta'] = localTheta['theta'] - baseTheta
    localTheta['cosine'] = np.cos(localTheta['projTheta']).astype(float)
    localTheta['sine'] = np.sin(localTheta['projTheta']).astype(float)

    layer_0 = []
    layer_1 = []
    layer_2 = []
    layer_3 = []
    # 0:same, 1:opposite, 2: tangent(up), 3:tangent(down)
    i = 0
    for th in localTheta['cosine'].values:
        Id = localTheta['LINK_ID'].values[i]
        if th > bd1:
            layer_0.append(Id)
        elif th < -bd1:
            layer_1.append(Id)
        elif abs(th) < bd2:
            if localTheta['sine'].values[i] > 0:
                layer_2.append(Id)
            else:
                layer_3.append(Id)
        i += 1

    return (np.array([layer_0, layer_1, layer_2, layer_3]))


'''
inout1 (14, 24, 34) / inout2 (15, 25, 35) --> tangent_opt = False
'''

def filterInflowIds(startDF, endDF, targetId, Id, tangent_opt=True):
    st = startDF[startDF['LINK_ID'] == targetId][['start_x', 'start_y']].values[0]
    ed = endDF[endDF['LINK_ID'] == targetId][['end_x', 'end_y']].values[0]
    vec = ed - st

    x1 = startDF[startDF['LINK_ID'] == Id]['start_x'].values[0]
    y1 = startDF[startDF['LINK_ID'] == Id]['start_y'].values[0]
    x2 = endDF[endDF['LINK_ID'] == Id]['end_x'].values[0]
    y2 = endDF[endDF['LINK_ID'] == Id]['end_y'].values[0]

    if tangent_opt == True:
        det = vec[0] * x2 + vec[1] * y2 - ed[0] * vec[0] - ed[1] * vec[1]
    else:
        det = -1

    if det < 0:
        st_dist = abs(x2 * vec[1] + y2 * vec[0]) / np.sqrt(sum(vec ** 2))
        ed_dist = abs(x1 * vec[1] + y1 * vec[0]) / np.sqrt(sum(vec ** 2))

        if st_dist > ed_dist:
            return Id

def filterOutflowIds(startDF, endDF, targetId, Id, tangent_opt=True):
    st = startDF[startDF['LINK_ID'] == targetId][['start_x', 'start_y']].values[0]
    ed = endDF[endDF['LINK_ID'] == targetId][['end_x', 'end_y']].values[0]
    vec = ed - st

    x1 = startDF[startDF['LINK_ID'] == Id]['start_x'].values[0]
    y1 = startDF[startDF['LINK_ID'] == Id]['start_y'].values[0]
    x2 = endDF[endDF['LINK_ID'] == Id]['end_x'].values[0]
    y2 = endDF[endDF['LINK_ID'] == Id]['end_y'].values[0]

    if tangent_opt == True:
        det = vec[0] * x1 + vec[1] * y1 - st[0] * vec[0] - st[1] * vec[1]
    else:
        det = 1

    if det > 0:
        st_dist = abs(x2 * vec[1] + y2 * vec[0]) / np.sqrt(sum(vec ** 2))
        ed_dist = abs(x1 * vec[1] + y1 * vec[0]) / np.sqrt(sum(vec ** 2))

        if st_dist < ed_dist:
            return Id


def layerIds_inout(targetId, thetaDF, localIds, startDF, endDF, theta_bd=15, tangent_opt=True):
    bd = np.cos((theta_bd * pi / 180))

    baseTheta = float(thetaDF[thetaDF['LINK_ID'] == targetId]['theta'].values[0])
    localTheta = thetaDF[thetaDF['LINK_ID'].isin(localIds)]
    localTheta['projTheta'] = (localTheta['theta'] - baseTheta).astype(float)
    localTheta['cosine'] = (np.cos(localTheta['projTheta'])).astype(float)

    # 0:same, 1:opposite, 2:in, 3:out
    layer_0 = []
    layer_1 = []
    layer_2 = []
    layer_3 = []

    i = 0
    for th in localTheta['cosine'].values:
        Id = localTheta['LINK_ID'].values[i]

        if th > bd:
            layer_0.append(Id)
        elif th < -bd:
            layer_1.append(Id)
        else:
            inId = filterInflowIds(startDF, endDF, targetId, Id, tangent_opt)
            outId = filterOutflowIds(startDF, endDF, targetId, Id, tangent_opt)
            if inId is not None:
                layer_2.append(inId)
            if outId is not None:
                layer_3.append(outId)
        i += 1

    return np.array([layer_0, layer_1, layer_2, layer_3])


'''
road connection based on start/end points (17, 27, 37) / (18, 28, 38)
'''
def layerIds_connect(targetId, localIds, linkDF, k_hop = 1):
    targetStart = linkDF[linkDF['LINK_ID'] == targetId]['START'].values[0]
    targetEnd = linkDF[linkDF['LINK_ID'] == targetId]['END'].values[0]
    localDF = linkDF[linkDF['LINK_ID'].isin(localIds)]

    neighbors_1hop = localDF[((localDF['END'] == targetStart) | (localDF['START'] == targetEnd)) & (((localDF['END'] != targetStart) | (localDF['START'] != targetEnd)))]['LINK_ID'].values.tolist()
    if k_hop == 1:
        return neighbors_1hop
    elif k_hop == 2:
        neighbors_2hop = []
        for nei in neighbors_1hop:
            START = localDF[localDF['LINK_ID'] == nei]['START'].values[0]
            END = localDF[localDF['LINK_ID'] == nei]['END'].values[0]
            neighbors_2hop += (localDF[((localDF['END'] == START) | (localDF['START'] == END)) & (((localDF['END'] != START) | (localDF['START'] != END)))])['LINK_ID'].values.tolist()
        neighbors_2hop = np.unique(neighbors_2hop)
        return np.append(neighbors_1hop, neighbors_2hop)


def saveLayerIds(links, localIds, data_type, th, traj_opt='all', r=10):
    theta = pd.DataFrame(np.load('data/thetas.npy'))
    theta.columns = ['LINK_ID', 'theta']

    layerIds = {}
    st = tt.time()
    for l in links:
        if data_type % 10 == 3:
            layerIds.update({l: layerIds_NEWS(l, theta, localIds[l], theta_bd=th)})
        elif data_type % 10 == 4:
            df = pd.read_csv('data/theta.csv', index_col=0)
            start_xy = df[['LINK_ID', 'start_x', 'start_y']]
            end_xy = df[['LINK_ID', 'end_x', 'end_y']]
            layerIds.update({l: layerIds_inout(l, theta, localIds[l], start_xy, end_xy, theta_bd=th, tangent_opt=True)})
        elif data_type % 10 == 5:
            df = pd.read_csv('data/theta.csv', index_col=0)
            start_xy = df[['LINK_ID', 'start_x', 'start_y']]
            end_xy = df[['LINK_ID', 'end_x', 'end_y']]
            layerIds.update({l: layerIds_inout(l, theta, localIds[l], start_xy, end_xy, theta_bd=th, tangent_opt=False)})
        # elif data_type % 10 == 7:
        #     layerIds.update({l: layerIds_connect(l, localIds[l], link_details, k_hop=1)})
        # elif data_type % 10 == 8:
        #     layerIds.update({l: layerIds_connect(l, localIds[l], link_details, k_hop=2)})

        if len(layerIds) % 100 == 0:
            print(len(layerIds))
            print(tt.time() - st)
            st = tt.time()

    np.save('data/layerIds_' + str(traj_opt) + '/' + str(data_type) + '/layerIds_img' + str(2*r+1) + '.npy', layerIds)


th = 15
for r in [2, 5, 20, 50]:
    for data_type in [23, 24, 25, 33, 34, 35]:
        for traj_opt in ['all', 'known']:
            if data_type > 30:
                links = np.load('data/linkIds_cx2.npy')
                localIds = np.load('data/localIds/localIds_all_cx2_R' + str(r) + '.npy').item()
            elif data_type > 20:
                links = np.load('data/linkIds_cx1.npy')
                localIds = np.load('data/localIds/localIds_all_cx1_R' + str(r) + '.npy').item()
            # else:
            #     links = np.load('data/linkIds_hw.npy')
            #     localIds = np.load('data/localIds/localIds_hw_R' + str(r) + '.npy').item()

            st = tt.time()
            saveLayerIds(links, localIds, data_type, th, traj_opt, r)
            print("###########################################")
            print r
            print 'Type %i is completed. (TIME: %i seconds)' %(data_type, tt.time()-st)