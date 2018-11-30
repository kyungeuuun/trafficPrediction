import numpy as np
import pandas as pd
from workalendar.asia import SouthKorea

dates = pd.date_range(start='2014/01/01 00:00:00', end='2018/11/01 00:00:00', closed='left', freq='1H')
weekdays = np.array(dates.weekday_name.tolist())
# print weekdays
# print dates.weekday_name.tolist()

cal = SouthKorea()
holidays = cal.holidays(2014) + cal.holidays(2015) + cal.holidays(2016) + cal.holidays(2017) + cal.holidays(2018)
result = []
for i in range(len(holidays)):
    result.append(holidays[i][0])


date_idx = np.zeros(len(dates), dtype=float)
date_idx[np.where(weekdays == 'Saturday')[0]] += 1
date_idx[np.where(weekdays == 'Sunday')[0]] += 1
date_idx[np.where(dates.isin(result))[0]] += 100

print len(pd.date_range(start='2014/01/01 00:00:00', end='2018/01/01 00:00:00', closed='left', freq='1H')) #35064
print len(pd.date_range(start='2014/01/01 00:00:00', end='2014/04/01 00:00:00', closed='left', freq='1H')) #2160
print len(pd.date_range(start='2018/01/01 00:00:00', end='2018/11/01 00:00:00', closed='left', freq='1H')) #7296
print len(pd.date_range(start='2018/01/01 00:00:00', end='2018/04/01 00:00:00', closed='left', freq='1H')) #2160

np.save('data/train_dateIdx.npy', [i for i in np.where(date_idx[:35064] == 0)[0] if i > 2160])
np.save('data/test_dateIdx.npy', [i for i in np.where(date_idx[35064:] == 0)[0] if i > 2160])