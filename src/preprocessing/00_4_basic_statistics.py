import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spd = np.load('data/spd_interpolated.npy').item()
links = np.load('data/all_links_afterFiltering2.npy')

times = pd.date_range(start='1/1/2014', end='11/1/2018', closed='left', freq='1H')

timeIndex = pd.DataFrame({'timestamp': times, 'year': [time.year for time in times],
                          'month': [time.month for time in times], 'day': [time.day for time in times],
                          'hour': [time.hour for time in times], 'weekday': [time.weekday() for time in times]})
# weekday: monday==0 ~ sunday==6

normalIdx_1 = timeIndex[(timeIndex['year'] == 2017) & (timeIndex['month'] == 12) & (timeIndex['day'] == 11)].index
normalIdx_2 = timeIndex[(timeIndex['year'] == 2017) & (timeIndex['month'] == 12) & (timeIndex['day'] == 18)].index
normalIdx_3 = timeIndex[(timeIndex['year'] == 2017) & (timeIndex['month'] == 12) & (timeIndex['day'] == 25)].index
christmasIdx = timeIndex[(timeIndex['year'] == 2018) & (timeIndex['month'] == 1) & (timeIndex['day'] == 1)].index

# normalIdx_1 = timeIndex[(timeIndex['year'] == 2014) & (timeIndex['month'] == 12) & (timeIndex['day'] == 11)].index
# normalIdx_2 = timeIndex[(timeIndex['year'] == 2014) & (timeIndex['month'] == 12) & (timeIndex['day'] == 18)].index
# normalIdx_3 = timeIndex[(timeIndex['year'] == 2014) & (timeIndex['month'] == 12) & (timeIndex['day'] == 25)].index
# christmasIdx = timeIndex[(timeIndex['year'] == 2015) & (timeIndex['month'] == 1) & (timeIndex['day'] == 1)].index

print normalIdx_1[0] / 24
print normalIdx_2[0] / 24
print normalIdx_3[0] / 24
print christmasIdx

# for l in links[116:117]:
    # print spd[l][normalIdx_1]
    # print spd[l][normalIdx_2]
    # print spd[l][normalIdx_3]
    # print spd[l][christmasIdx]
    #
    # plt.plot(spd[l][normalIdx_1], c='b')
    # plt.plot(spd[l][normalIdx_2], c='k')
    # plt.plot(spd[l][normalIdx_3], c='g')
    # plt.plot(spd[l][christmasIdx], c='r')
    # plt.show()


for day in [1440, 1447, 1454, 1461]: #[344, 351, 358, 365]:
    ldata = spd[links[116]].reshape([-1, 24])
    plt.plot(ldata[day, :])
plt.show()