import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

links = np.load('data/all_links_afterFiltering2.npy')
spd_bf = np.load('data/spd_augmented_minusWhenAllZero.npy').item()
spd = np.load('data/spd_interpolated_ignoreErrTimes.npy').item()
# spd_array = []
# for l in links:
#     spd_array.append(spd[l])
# np.save('data/spdArray.npy', np.array(spd_array))
spd_array = np.load('data/spdArray.npy')
coords = pd.read_csv('rawdata/LINK_VERTEX_seoulonly.csv', index_col=0)

#before interpolation
# count_zero = []
# for l in links:
#     count_zero.append(sum(spd_bf[l] <= 0))
#
#     if len(count_zero) % 10 == 0:
#         print np.mean(count_zero)

#after interpolation
# count_zero = []
# for l in links:
#     count_zero.append(sum(spd[l] == 0))
# print sum(count_zero) #0


# print sum(sum(spd_array == -1)) #185710
nonzeroIdx = np.where(spd_array[0] > 0)[0]
# print np.mean(np.mean(spd_array[:, nonzeroIdx]))
# print np.std(spd_array[:, nonzeroIdx].tolist())
# print np.std(np.mean(spd_array[:, nonzeroIdx], axis=0))
# print np.std(np.mean(spd_array[:, nonzeroIdx], axis=1))
# print np.mean(np.std(spd_array[:, nonzeroIdx], axis=0))
# print np.mean(np.std(spd_array[:, nonzeroIdx], axis=1))
#
# 27.383533355212318
# 10.906469816599154
# 4.711240523374501
# 8.248286025414014
# 9.689213866037544
# 6.7267516728409165

# linkStd = np.std(spd_array[:, nonzeroIdx], axis=0)
# p = pd.DataFrame(columns=('x', 'y', 'std'))
# i = 0
# for i in range(len(links)):
#     linkId = links[i]
#     std = linkStd[i]
#     coord = coords[coords['LINK_ID'] == linkId].sort_values('VER_SEQ')[['GRS80TM_X', 'GRS80TM_Y']].values
#
#     for c in range(coord.shape[0]):
#         p.loc[i] = [coord[c, 0], coord[c, 1], std]
#         i += 1
#
# plt.scatter(p['x'], p['y'], s=2, c=p['std'])
# plt.colorbar()
# plt.show()


def perfMetrics(err, truth):
    mae = np.mean(np.abs(err))
    mape = np.true_divide(np.abs(err), truth)
    mape = mape[~np.isnan(mape)]
    mape = np.mean(mape)
    rmse = np.sqrt(np.mean(np.square(err)))

    return (mape, mae, rmse)
#
# #ha_week
# print 'week'
# err = []
# truth = []
# for l in range(len(links)):
#     dt = spd_array[l, :]
#     for t in range(24*7+1, len(dt)):
#         if dt[t] >= 0:
#             err.append(dt[t-24*7] - dt[t])
#             truth.append(dt[t])
#
# print perfMetrics(err, truth)
#
# #ha_day
# print 'day'
# err = []
# truth = []
# for l in range(len(links)):
#     dt = spd_array[l, :]
#     for t in range(24*7+1, len(dt)):
#         if dt[t] >= 0:
#             err.append(dt[t-24] - dt[t])
#             truth.append(dt[t])
#
# print perfMetrics(err, truth)
#
# #ha_hour
# print 'hour'
# err = []
# truth = []
# for l in range(len(links)):
#     dt = spd_array[l, :]
#     for t in range(24*7+1, len(dt)):
#         if dt[t] >= 0:
#             err.append(dt[t-1] - dt[t])
#             truth.append(dt[t])
#
# print perfMetrics(err, truth)
#
#ha_month
print 'month'
err = []
truth = []
for l in range(len(links)):
# for l in range(1):
    dt = spd_array[l, :]
    for t in range(24*7+1, len(dt)):
        if dt[t] >= 0:
            ha_range = range((t-24*7), max(0, t-24*7*4-1), -(24*7))
            # print len(ha_range)
            err.append(np.mean(dt[ha_range]) - dt[t])
            truth.append(dt[t])

print perfMetrics(err, truth)


#ha: all
print 'all'
err = []
truth = []
for l in range(len(links)):
    dt = spd_array[l, :]
    for t in range(24*7+1, len(dt)):
        if dt[t] >= 0:
            ha_range = range((t-24*7), 0, -(24*7))
            err.append(np.mean(dt[ha_range]) - dt[t])
            truth.append(dt[t])
print perfMetrics(err, truth)