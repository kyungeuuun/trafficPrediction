import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Speed interpolation: linear interpolation for the zero_valued cells
'''

spd = np.load('data/spd_augmented_minusWhenAllZero.npy').item()
links = np.load('data/all_links_afterFiltering2.npy')
times = pd.date_range(start='1/1/2014', end='11/1/2018', closed='left', freq='1H')

# print spd[1000017200]
# print spd[1000017100]

plt.plot(times[17300:17800], spd[1000017200][17300:17800])
# plt.plot(spd[1000017200][-42000:-41900])
plt.show()
#
# plt.plot(spd[1000017100][:100])
# plt.plot(spd[1000017100][-100:])
# plt.show()
#
# plt.plot(spd[1000015200][:100])
# plt.plot(spd[1000015200][-100:])
# plt.show()
#
# plt.plot(spd[1000015100][:100])
# plt.plot(spd[1000015100][-100:])
# plt.show()


# i = 0
# st = time.time()
#
# spd_interp = {}
# for l in links:
#     lspd = np.array(spd[l])
#     interp_t = np.where(lspd == 0)[0]
#     normal_t = np.where(lspd > 0)[0]
#
#     if len(interp_t) > 0:
#         if len(interp_t) > 1000:
#             plt.figure(figsize = (200, 15))
#             plt.plot(lspd, c='b')
#
#             lspd[interp_t] = np.interp(interp_t, normal_t, lspd[normal_t])
#             print l
#             plt.plot(lspd, c='r')
#             plt.savefig('/data2/keun/traffic_prediction/checkImages/spd_interpolation/' + str(l) + '_' +
#                         str(len(interp_t)) + '.png', dpi=244)
#             plt.close()
#         else:
#             lspd[interp_t] = np.interp(interp_t, normal_t, lspd[normal_t])
#
#     spd_interp.update({l: lspd})
#     i += 1
#
#     if i % 1000 == 0:
#         print i
#         print time.time() - st
#
# np.save('data/spd_interpolated_ignoreErrTimes.npy', spd_interp)