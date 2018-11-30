import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

links = np.load('data/all_links_afterFiltering2.npy')
spd = np.load('data/spd_interpolated_ignoreErrTimes.npy').item()
spd_array = np.load('data/spdArray.npy')
coords = pd.read_csv('rawdata/LINK_VERTEX_seoulonly.csv', index_col=0)
coords = coords[coords['LINK_ID'].isin(links)]
info = pd.read_csv('rawdata/link_information.csv')
info = info[info['LINK_ID'].isin(links)]

# dt = info['LINK_ID'].values
# for l in links:
#     if l not in dt:
#         print l
# all data is in info file

p = coords[(coords['GRS80TM_X'] > 203000) & (coords['GRS80TM_X'] < 210000) & (coords['GRS80TM_Y'] > 442500) & (coords['GRS80TM_Y'] < 449500)]
plt.figure(figsize=(10,10))
plt.scatter(p['GRS80TM_X'], p['GRS80TM_Y'], s=1)
plt.savefig('images/00_7_dataScope_gangnam_cx1.png', dpi=244)
# plt.show()
print len(np.unique(p['LINK_ID'])) #457
cx1_links = np.unique(p['LINK_ID'])
np.save('data/linkIds_cx1.npy', cx1_links)

p = coords[(coords['GRS80TM_X'] > 187000) & (coords['GRS80TM_X'] < 194000) & (coords['GRS80TM_Y'] > 444000) & (coords['GRS80TM_Y'] < 451000)]
plt.figure(figsize=(10,10))
plt.scatter(p['GRS80TM_X'], p['GRS80TM_Y'], s=1)
plt.savefig('images/00_7_dataScope_guro_cx2.png', dpi=244)
# plt.show()
print len(np.unique(p['LINK_ID'])) #589
cx2_links = np.unique(p['LINK_ID'])
np.save('data/linkIds_cx2.npy', cx2_links)