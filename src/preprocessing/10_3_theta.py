import numpy as np
import pandas as pd
import math

coord = pd.read_csv('rawdata/LINK_VERTEX_seoulonly.csv', index_col=0)
links = np.load('data/all_links_afterFiltering2.npy')
coord = coord[coord['LINK_ID'].isin(links)]

min_x = min(coord['GRS80TM_X'])
min_y = min(coord['GRS80TM_Y'])
# print min_x #182071.29
# print min_y #437710.11
coord['x'] = ((coord['GRS80TM_X'] - min_x) / 100).astype(int)
coord['y'] = ((coord['GRS80TM_Y'] - min_y) / 100).astype(int)

linkInfo = pd.read_csv('rawdata/link_information.csv')

pi = math.pi
coordDF = pd.DataFrame(columns = ('LINK_ID', 'start_x', 'start_y', 'end_x', 'end_y', 'length', 'theta'))
coordDict = {}
i = 0
for l in links:
    if l in linkInfo['LINK_ID']:
        if (linkInfo[linkInfo['LINK_ID'] == l]['coordID'].values == l):
            st = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index().head(1)
            ed = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index().tail(1)
        else:
            l = linkInfo[linkInfo['LINK_ID'] == l]['coordID'].values[0]
            st = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index().tail(1)
            ed = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index().head(1)
    else:
        st = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index().head(1)
        ed = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index().tail(1)

    vec = (st[['GRS80TM_X', 'GRS80TM_Y']].values - ed[['GRS80TM_X', 'GRS80TM_Y']].values)[0]
    dist = np.linalg.norm(vec)

    if vec[1] >= 0:
        theta = np.arccos(vec[0] / dist)
    else:
        theta = 2*pi - np.arccos(vec[0] / dist)

    coordDF.loc[i] = [l, st[['x', 'y']].values[0][0], st[['x', 'y']].values[0][1], ed[['x', 'y']].values[0][0], ed[['x', 'y']].values[0][1], dist, theta]
    coordDict.update({l: theta})

    i += 1

coordDF.to_csv('data/notUsedInModel/theta.csv')
np.save('data/thetas.npy', coordDF[['LINK_ID', 'theta']])
np.save('data/thetaDict.npy', coordDict)