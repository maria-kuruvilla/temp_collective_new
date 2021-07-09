"""
Goal - To produce figure of water temperature data from Meramec river basin
June 15th 2021
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime
import matplotlib.dates as mdates

data = pd.read_csv('../../data/temp_collective/roi/water_data.csv')
new_header= data.iloc[1] 
data = data[2:]
data.columns = new_header 

dates = data.Date
# x_values = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in dates]
# y_values = data.avg

data = data.astype({'avg':'float64'})

data = data.astype({'stdev':'float64'})

# ax = plt.gca()



#ax.xaxis.set_major_locator(locator)

plt.close('all')
fs = 25

#fig = plt.figure(figsize=(16,10))
#fig, ax = plt.subplots(1)
ax = data.plot(x="Date", y="avg",legend = False, color = 'black', label = 'Average temperature', lw = 2)
#ax.plot(data.Date,data.avg)
# fig.autofmt_xdate()
# ax.fmt_xdata = mdates.DateFormatter('%Y-%m')
ax.fill_between(data.Date, data.avg - data.stdev, data.avg + data.stdev,alpha = 0.5, color = 'black', linewidth = 0)
ax.axvline('5/1/2018', label = 'Estimated start of \nspawing for Golden shiners', color = 'red', lw = 2)
plt.xlabel('Date', size = fs)
plt.ylabel('Water temperature '+r'($^{\circ}$C)', size = fs)
plt.legend(fontsize=fs, loc='upper left', framealpha = 0.5)

#plt.annotate(s='Estimated start spawing for Golden shiners', xy=(4000,27), fontsize = fs)
#formatter = mdates.DateFormatter("%Y-%m-%d")

# locator = mdates.MonthLocator()
# formatter = mdates.AutoDateFormatter(locator)

# ax.xaxis.set_major_formatter(formatter)


# ax.xaxis.set_major_locator(locator)

# date_form = mdates.DateFormatter("%m-%d")
# ax.xaxis.set_major_formatter(date_form)

# # Ensure a major tick for each week using (interval=1) 
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
#ax.set_title('Water temperature for Meramec river basin 2017-2018', fontsize = fs)
plt.xticks(rotation=20, fontsize = fs)
plt.yticks(fontsize = fs)
#ax.xaxis.set_major_locator(locator)
out_dir = '../../output/temp_collective/roi_figures/water_temp.png'
#fig.savefig(out_dir, dpi = 300)
plt.show()