import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time as tt

coord = pd.read_csv('rawdata/LINK_VERTEX_seoulonly.csv', index_col=0)

# cx1_links = np.load('data/linkIds_cx1.npy')
# cx2_links = np.load('data/linkIds_cx2.npy')
# links = np.unique(cx1_links + cx2_links)
links = np.load('data/all_links_afterFiltering2.npy')
coord = coord[coord['LINK_ID'].isin(links)]

min_x = min(coord['GRS80TM_X'])
min_y = min(coord['GRS80TM_Y'])
# print min_x #182071.29
# print min_y #437710.11

coord['x'] = ((coord['GRS80TM_X'] - min_x) / 100).astype(int)
coord['y'] = ((coord['GRS80TM_Y'] - min_y) / 100).astype(int)

def findTrajectories(start, end, interval=0.01):
    dist = np.linalg.norm(start - end)
    traj = []
    for v in np.arange(0, dist, interval):
        traj += [(start + ((end - start) * v / dist)).astype(int)]
    traj += [start.astype(int), end.astype(int)]
    return np.unique(traj, axis = 0)

allTraj = {}
knownTraj = {}
st = tt.time()
for l in links:
    knownCoords = coord[coord['LINK_ID'] == l].sort_values('VER_SEQ').reset_index()[['x', 'y']].values
    if len(knownCoords) > 1:
        traj = []
        for c in range(len(knownCoords) - 1):
            traj.append(findTrajectories(knownCoords[c], knownCoords[c+1]))
        traj = np.unique([x for y in traj for x in y], axis=0)
    else:
        print l
    allTraj.update({l: traj})
    knownTraj.update({l: knownCoords})

    if len(allTraj) % 100 == 0:
        print len(allTraj)
        print tt.time() - st

np.save('data/knownTraj.npy', knownTraj)
np.save('data/allTraj.npy', allTraj)
