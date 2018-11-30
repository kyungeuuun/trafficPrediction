import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time as tt

coord = pd.read_csv('rawdata/LINK_VERTEX_seoulonly.csv', index_col=0)

cx1_links = np.load('data/linkIds_cx1.npy')
cx2_links = np.load('data/linkIds_cx2.npy')
# links = np.unique(cx1_links + cx2_links)
links = np.load('data/all_links_afterFiltering2.npy')
coord = coord[coord['LINK_ID'].isin(links)]

min_x = min(coord['GRS80TM_X'])
min_y = min(coord['GRS80TM_Y'])
# print min_x #182071.29
# print min_y #437710.11

coord['x'] = ((coord['GRS80TM_X'] - min_x) / 100).astype(int)
coord['y'] = ((coord['GRS80TM_Y'] - min_y) / 100).astype(int)
# print max(coord['x']) #340
# print max(coord['y']) #275

knownTraj = np.load('data/knownTraj.npy').item()
allTraj = np.load('data/allTraj.npy').item()

def plotKnownAndConnectedCoords_scatter(links, knownCoordDF, allTraj, filename):
    plt.figure(figsize=(20, 20))
    plt.scatter(allTraj[links[0]][:, 0], allTraj[links[0]][:, 1], c='b', s=5, label='Connected')
    for l in links:
        plt.scatter(allTraj[l][:, 0], allTraj[l][:, 1], c='b', s=5)
    knownCoords = knownCoordDF[knownCoordDF['LINK_ID'].isin(links)].sort_values('VER_SEQ').reset_index()[['x', 'y']].values
    plt.scatter(knownCoords[:, 0], knownCoords[:, 1], c='r', s=5, label='Known')
    plt.legend(loc='best')
    plt.savefig(filename, dpi=244)
    plt.show()
    return

# plotKnownAndConnectedCoords_scatter(cx1_links, coord, allTraj, 'images/10_2_coordinates_cx1.png')
# plotKnownAndConnectedCoords_scatter(cx2_links, coord, allTraj, 'images/10_2_coordinates_cx2.png')

def plotKnownAndConnectedCoords_grid(link_type, links, traj, filename):
    if link_type == 'cx1':
        min_x = int((203000 - min(coord['GRS80TM_X'])) / 100)
        min_y = int((442500 - min(coord['GRS80TM_Y'])) / 100)
    elif link_type == 'cx2':
        min_x = int((187000 - min(coord['GRS80TM_X'])) / 100)
        min_y = int((444000 - min(coord['GRS80TM_Y'])) / 100)

    width_x = 70
    width_y = 70

    grid = np.zeros([width_x, width_y], dtype=float)

    for l in links:
        link_coords = traj[l]

        for c in link_coords:
            if (c[0] > min_x) & (c[0] < (min_x + width_x)) & (c[1] > min_y) & (c[1] < (min_y + width_y)) & (
                (c[0] - min_x) < width_x) & ((c[1] - min_y) < width_y):
                if grid[c[1] - min_y, c[0] - min_x] == 0:
                    grid[c[1] - min_y, c[0] - min_x] += 1

    plt.figure(figsize=(20, 20))
    plt.imshow(grid, cmap=plt.get_cmap('magma'))
    plt.legend(loc='best')
    plt.xlim(0, width_y)
    plt.ylim(0, width_x)
    plt.savefig(filename, dpi=244)
    plt.show()
    return

# plotKnownAndConnectedCoords_grid('cx1', cx1_links, knownTraj, 'images/10_2_knownTraj_cx1.png')
# plotKnownAndConnectedCoords_grid('cx1', cx1_links, allTraj, 'images/10_2_allTraj_cx1.png')
plotKnownAndConnectedCoords_grid('cx2', cx2_links, knownTraj, 'images/10_2_knownTraj_cx2.png')
plotKnownAndConnectedCoords_grid('cx2', cx2_links, allTraj, 'images/10_2_allTraj_cx2.png')