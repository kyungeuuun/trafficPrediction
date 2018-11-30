import numpy as np

allTraj = np.load('data/allTraj.npy').item()
knownTraj = np.load('data/knownTraj.npy').item()

cx1_links = np.load('data/linkIds_cx1.npy')
cx2_links = np.load('data/linkIds_cx2.npy')

def findLocalIds(links, coordDict, img_radius=10):
    localIds = {}
    for l in links:
        (minX, minY) = np.min(coordDict[l], axis=0) - img_radius
        (maxX, maxY) = np.max(coordDict[l], axis=0) + img_radius

        result = []
        for k in links:
            detX = sum((minX < coordDict[k][:, 0]) & (coordDict[k][:, 0] < maxX))
            detY = sum((minY < coordDict[k][:, 1]) & (coordDict[k][:, 1] < maxY))
            if (detX > 0) & (detY > 0):
                result.append(k)
        localIds.update({l: result})

    return localIds

# r = 10
for r in [2, 5, 20, 50]:
    np.save('data/localIds/localIds_all_cx1_R' + str(r) + '.npy', findLocalIds(cx1_links, allTraj, r))
    np.save('data/localIds/localIds_all_cx2_R' + str(r) + '.npy', findLocalIds(cx2_links, allTraj, r))
    np.save('data/localIds/localIds_known_cx1_R' + str(r) + '.npy', findLocalIds(cx1_links, knownTraj, r))
    np.save('data/localIds/localIds_known_cx2_R' + str(r) + '.npy', findLocalIds(cx2_links, knownTraj, r))