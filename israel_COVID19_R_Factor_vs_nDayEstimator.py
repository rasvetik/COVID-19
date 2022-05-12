#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:20:48 2021

@author: Doron Bar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import time

eps = 2**(-10)
last_day = time.strftime("%d%m%Y")
base_db = pd.read_csv('Israel_db.csv')

base_country = 'Israel'

r_db = pd.read_csv('israel_COVID19_R_Factor.csv')

israel_data = base_db.copy()
israel_data = israel_data.sort_values(by='Date')
dates = pd.to_datetime(israel_data.Date)

smooth_factor = [4, 7, 14]
line_width = [1, 1, 3]
smooth_colors = ('g', 'b', 'r', 'm')

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
d0 = 0
dr = 427    # until 1.5.21
d1 = 9 + dr
count = -1
for w in smooth_factor:
    count += 1
    z = list([])
    y = list([])
    for i in range(w+5, len(dates)):
        p00 = i-w
        p01 = i
        try:
            if israel_data.Recovered.iloc[p00] == israel_data.Recovered.iloc[p01-1]:
                z1 = [0, 0]
            else:
                z1 = np.polyfit(israel_data.Recovered.iloc[p00:p01], israel_data.Deaths.iloc[p00:p01], 1, eps)
        except:
            z1 = [0, 0]
        z.append(max(z1[0], 0))
        y.append(z1[1])
    ax1.plot(dates[d0 + w + 5:d1], z[d0:d1 - w - 5], linewidth=line_width[count], color=smooth_colors[count])

ax2.plot(dates[9:d1], r_db.rFactor[:dr], linewidth=line_width[count], color=smooth_colors[1])
ax2.plot(dates[9:d1], [1] * len(dates[9:d1]), '--', linewidth=2, color='k')
ax1.grid()
ax2.grid()
leg = ('$n = ' + str(smooth_factor[0]) + '$', '$n = ' + str(smooth_factor[1]) + '$',
       '$n = ' + str(smooth_factor[2]) + '$')
ax1.legend(leg)
ax2.legend(('R in Israel', 'R = 1'))
ax1.set_ylim([0, 0.08])
ax1.set_ylabel('$\epsilon_n(t)$', fontsize=20, fontweight='bold')
ax2.set_ylabel('Infection coefficient R', fontsize=18, fontweight='bold')
ax1.set_title(base_country + ' - ' +
              'nDayEstimator $\epsilon_n(t)$ vs infection coefficient R, March 2020 to May 2021', fontsize=16)
save_string = 'nDayEstimator_vs_R_1.3.20to1.5.21.png'
print(save_string)
f.savefig(os.path.join(os.getcwd(), last_day, save_string))
