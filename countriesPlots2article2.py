#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:20:08 2021
@author: Doron Bar
"""
import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
from Utils import *


######################################################################################################################
# Plot: 1,2) n-days-estimators, 3) infected & confirmed, 4) recovered & death
def plot8subplots(given_data, last_day, base_country, plot2march=False):
    given_data = given_data.sort_values(by='Date')
    dates = pd.to_datetime(given_data.Date)
    
    data1 = given_data.NewDeaths
    data2 = given_data.NewConfirmed
    data3 = given_data.NewRecovered
    
    data1smo = smooth_vector(data1)
    data2smo = smooth_vector(data2)
    data3smo = smooth_vector(data3)
    
    # num of recovered at first deaths
    deaths = given_data.Deaths
    deaths_vector = deaths.to_numpy()
    nonzero_death_idx = np.where(deaths_vector > 0)[0][0]
    print(' First Deaths = ' + str(deaths_vector[nonzero_death_idx]) + ', with Recovered = ' +
          str(int(given_data.Recovered[nonzero_death_idx])))
    zero_recovered = False
    if given_data.Recovered[nonzero_death_idx] == 0:
        zero_recovered = True

    # ratio plots
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 18))
    
    d0 = 0
    d1 = 170
    d2 = 155
    d3 = len(dates)
    if plot2march:
        d3 = 374  # 2021-03-01
    
    smooth_factor = [4, 7, 14]
    smooth_colors = ('g', 'b', 'r', 'm')
    leg = ('$n = ' + str(smooth_factor[0]) + '$', '$n = ' + str(smooth_factor[1]) + '$', '$n = ' + str(smooth_factor[2]) + '$')
    
    count = -1
    for w in smooth_factor:
        count += 1
        z = list([])
        y = list([])
        yi = list([])
        for i in range(w+5, len(dates)):
            p00 = i-w
            p01 = i
            try:
                z1 = [0, 0]
                if given_data.Recovered.iloc[p01-1] - given_data.Recovered.iloc[p00] > 0:
                    z1 = np.polyfit(given_data.Recovered.iloc[p00:p01], given_data.Deaths.iloc[p00:p01], 1)
            except:
                z1 = [0, 0]
            z.append(z1[0])
            y.append(z1[1])
            yi.append(data2smo[i] - (1+z1[0])*data3smo[i])
        
        ax1.plot(dates[d0 + w + 5:d1], z[d0:d1 - w - 5], linewidth=2, color=smooth_colors[count])
        ax2.plot(dates[d2:d3], z[d2 - w - 5:d3 - w - 5], linewidth=2, color=smooth_colors[count])
        ax3.plot(dates[d0 + w + 5:d1], yi[d0:d1 - w - 5], linewidth=2, color=smooth_colors[count])
        ax4.plot(dates[d2:d3], yi[d2 - w - 5:d3 - w - 5], linewidth=2, color=smooth_colors[count])

    yi = list([])
    for i in range(len(dates)):
        yi.append(data2smo[i] - data3smo[i] - data1smo[i])
    ax3.plot(dates[0:d1], yi[0:d1], linewidth=3, color='k')
    ax4.plot(dates[d2:d3], yi[d2:d3], linewidth=3, color='k')
    
    ax1.grid()
    ax1.legend(leg)
    ax1.set_ylabel('$\epsilon_n(t)$', fontsize=18, fontweight='bold')
    ax1.set_title(base_country + ' - ' + 'Death rate to recovery rate ratio', fontsize=16)
    ax2.grid()
    ax2.legend(leg, loc='best')
    
    ax3.grid()
    ax3.legend(leg)
    ax3.set_ylabel('$d_{0,n}(t)$', fontsize=18, fontweight='bold')
    
    ax4.grid()
    ax4.legend(leg, loc='best')
    
    ax7.plot(dates[d0 + smooth_factor[0] + 5:d1], smooth_vector(data3)[d0 + smooth_factor[0] + 5:d1], linewidth=2, color='k')
    ax7.set_ylabel('Daily recovered', fontsize=18, fontweight='bold')
    ax7.grid(axis='x')
    
    ax72 = ax7.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax72.plot(dates[d0 + smooth_factor[0] + 5:d1], smooth_vector(data1)[d0 + smooth_factor[0] + 5:d1], color=color, linewidth=2)
    ax72.tick_params(axis='y', labelcolor=color)
    
    ax5.plot(dates[d0 + smooth_factor[0] + 5:d1], given_data.Active[d0 + smooth_factor[0] + 5:d1], linewidth=2, color='k')
    ax5.set_ylabel('Infected', fontsize=18, fontweight='bold')  # Active
    ax5.grid(axis='x')
    
    ax52 = ax5.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax52.plot(dates[d0 + smooth_factor[0] + 5:d1], smooth_vector(data2)[d0 + smooth_factor[0] + 5:d1], color=color, linewidth=2)
    ax52.tick_params(axis='y', labelcolor=color)
    
    ax8.plot(dates[d2:d3], smooth_vector(data3)[d2:d3], linewidth=2, color='k')
    ax8.grid(axis='x')
    
    ax82 = ax8.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax82.set_ylabel('Daily deaths', color=color, fontsize=18, fontweight='bold')  # New Death
    ax82.plot(dates[d2:d3], smooth_vector(data1)[d2:d3], color=color, linewidth=2)
    ax82.tick_params(axis='y', labelcolor=color)
    
    ax6.plot(dates[d2:d3], given_data.Active[d2:d3], linewidth=2, color='k')
    ax6.grid(axis='x')
    
    ax62 = ax6.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax62.set_ylabel('Daily Confirmed', color=color, fontsize=18, fontweight='bold')
    ax62.plot(dates[d2:d3], smooth_vector(data2)[d2:d3], color=color, linewidth=2)
    ax62.tick_params(axis='y', labelcolor=color)
    
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax5.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax6.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax7.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax8.get_xticklabels(), rotation=30, ha='right')
    
    if True:
        save_string = last_day + '_D2Rratio_' + base_country + '_' + str(smooth_factor[0]) + '.' + \
                      str(smooth_factor[1]) + '.' + str(smooth_factor[2]) + '.png'
        print(save_string)
        f.savefig(os.path.join(os.getcwd(), last_day, save_string))
    
        plt.pause(.001)
        plt.close('all')
    
    return zero_recovered, dates, deaths_vector, nonzero_death_idx


######################################################################################################################
# Initial Parameters
last_day = time.strftime("%d%m%Y")
checkFirstDeath = True
plot2march = False

# Create file for saving countries with zero recovered while there are deaths already
if checkFirstDeath:
    path_to_file = os.path.join(os.getcwd(), last_day, 'firstCovid19Death.csv')
    fid = open(path_to_file, 'w')
    row = ['country', 'date', 'deaths', 'recovers']
    writer = csv.writer(fid)
    writer.writerow(row)

# Countries
run_all_countries = False
if run_all_countries:
    countries_files = [fn for fn in os.listdir(last_day) if fn[-7:] == '_db.csv']
else:
    countries_files = ['Israel_db.csv']

n = 0
# count countries with zero recovered while there are deaths already
for i in range(len(countries_files)):
    print(countries_files[i][:-7])
    base_db = pd.read_csv(os.path.join(os.getcwd(), last_day, countries_files[i]))
    base_country = countries_files[i][:-7]
    try:
        current_data = base_db.copy()
        zeroRecovered, date_vec, death_vec, ii = plot8subplots(current_data, last_day, base_country, plot2march)
        if zeroRecovered:
            n += 1
        if checkFirstDeath:
            row = [countries_files[i][:-7], str(date_vec[ii])[:10], str(death_vec[ii]), str(int(current_data.Recovered[ii]))]
            writer.writerow(row)
    except:
        print('No data')

print(str(n)+' countries with zero recovered while there are deaths already')

if checkFirstDeath:
    fid.close()
######################################################################################################################
