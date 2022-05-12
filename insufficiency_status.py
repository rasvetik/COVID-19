"""
Created on 9 Jan 2021
Sveta Raboy & Doron Bar

"""

import sys
from Utils import *
import numpy as np
import pandas as pd
import time
import os
import csv
import matplotlib.pyplot as plt
import warnings

plt.style.use('default')
warnings.filterwarnings("ignore")


######################################################################################################################
# Piecewise linear Polyfit By N
def piecewise_linear_polyfit_by_n(db, n=7):
    z = list([])
    data_len = len(db.Recovered)
    for i in range(n + 5, data_len):
        p00 = i - n
        p01 = i
        try:
            if db.Recovered.iloc[p00] == db.Recovered.iloc[p01-1]:
                z1 = [0, 0]
            else:
                z1 = np.polyfit(db.Recovered.iloc[p00:p01], db.Deaths.iloc[p00:p01], 1)
        except:
            z1 = [0, 0]
        z.append(z1[0])
    return z


######################################################################################################################
# Line Polyfit from length data till N
def line_polyfit_from_length_till_n(db, n=28):
    z = list([])
    data_len = len(db.Recovered)
    for i in range(n, data_len):
        p00 = data_len - i
        p01 = data_len
        try:
            if db.Recovered.iloc[p00] == db.Recovered.iloc[p01-1]:
                z1 = [0, 0]
            else:
                z1 = np.polyfit(db.Recovered.iloc[p00:], db.Deaths.iloc[p00:], 1)
        except:
            z1 = [0, 0]
        z.append(z1[0])
    return z


######################################################################################################################
# Line Polyfit to Data
def line_polyfit(db, p00=28, p01=None):
    if p01 is None:
        p01 = len(db.Recovered) - 1
    try:
        if db.Recovered.iloc[p00] == db.Recovered.iloc[p01-1]:
            z = [0, 0]
        else:
            z = np.polyfit(db.Recovered.iloc[p00:p01], db.Deaths.iloc[p00:p01], 1)
    except:
        z = [0, 0]
    return z


######################################################################################################################
# situation plot for country: variable cases
def situation_plot(data_db, base_country='Israel', p00=None, p01=None, daysM=28):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.grid(True)
    data_db['Date'] = pd.to_datetime(data_db['Date'])
    ax1.plot(data_db.Date, data_db.Confirmed, zorder=1, linewidth=3, color="blue", label="Confirmed")
    ax1.plot(data_db.Date, data_db.Recovered, zorder=1, linewidth=3, color="green", label="Recovered")
    ax1.plot(data_db.Date, data_db.Deaths, zorder=1, linewidth=3, color="red", label="Deceased")
    ax1.plot(data_db.Date, data_db.Active, zorder=1, linewidth=3, color="magenta", label="Active")
    ax1.set_title(base_country + ' - data from ' + str(data_db['Date'].iloc[-1])[:10], fontsize=16)
    ax1.set_ylim([-1, max(data_db['Confirmed']) + 20])
    ax1.legend()

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.fmt_xdata = formatter
    f.autofmt_xdate()
    # the other possibilities
    # ax1.xaxis.set_major_formatter(formatter)
    # weeks = mdates.WeekdayLocator()
    # weeks_fmt = mdates.DateFormatter('%Y-%m-%d')
    # ax1.xaxis.set_major_formatter(weeks_fmt)
    # ax1.xaxis.set_minor_locator(mdates.AutoDateLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax2.plot(data_db.Recovered, data_db.Deaths, zorder=1, linewidth=3, label=base_country)

    # daysM = 28
    lastM = line_polyfit_from_length_till_n(data_db, daysM)

    if p01 is None:
        p01 = len_data - 1
    if p00 is None:
        if len(lastM) == 0:
            p00 = len_data - daysM
        else:
            p00 = len_data - daysM - lastM.index(min(lastM))
    days = p01 - p00 + 1
    ok = True
    try:
        zM = line_polyfit(data_db, p00, p01)

        ax2.plot((0, data_db.Recovered.iloc[-1]), (zM[1], zM[1] + data_db.Recovered.iloc[-1] * zM[0]), '--',
                 linewidth=3, label='D ~ ' + str(int(zM[1])) + '+' + str(int(zM[0] * 1000) / 1000) + '*R')
        plt.plot(data_db.Recovered.iloc[p00], data_db.Deaths.iloc[p00], 'mo', zorder=1, markersize=12)
        plt.plot(data_db.Recovered.iloc[p01], data_db.Deaths.iloc[p01], 'mo', zorder=1, markersize=12)
    except:
        zM = [0, 0]
        print('No death vs. recovered ratio was found')
        ok = False

    ax2.set_title('D ~ ' + str(int(zM[1])) + '+' + str(int(zM[0] * 1000) / 1000) + '*R at the ' +
                  str(data_db.Date.iloc[p00].strftime('%d/%m/%y')) + ' for ' + str(days) + ' days', fontsize=16)
    ax2.set_xlabel('Recovered', fontsize=16)
    ax2.set_ylabel('Deaths', fontsize=16)
    ymin = data_db.Deaths.min()
    ymax = 1.05*data_db.Deaths.max()
    ax2.set_ylim(ymin, ymax)
    ax2.grid(True)
    print(base_country + ' - R: ' + str(data_db.Recovered.iloc[-1].astype(int)) + ', D: ' + str(data_db.Deaths.iloc[-1]))

    z5 = line_polyfit(data_db, -5)
    z7 = line_polyfit(data_db, -7)
    z14 = line_polyfit(data_db, -14)
    z28 = line_polyfit(data_db, -28)
    try:
        if z28.all() and z14.all() and z7.all() and z5.all():
            print(base_country + ' deaths vs. recovered''s ratio in last[28,14,7,5] days: [' + str(z28[0])[:6] + ',' +
                  str(z14[0])[:6] + ',' + str(z7[0])[:6] + ',' + str(z5[0])[:6] + ']')
        else:
            print(base_country + ' deaths vs. recovered''s ratio in last [28,14,7,5] days No Good')
            ok = False
    except:
        print(base_country + ' deaths vs. recovered''s ratio in last [28,14,7,5] days No Good')
        ok = False
    last_data_day = data_db.Date.max().strftime('%d%m%y')
    cur_day = time.strftime("%d%m%Y")
    save_string = last_data_day + '_situation_' + base_country + '_for_' + str(days) + '_days_till ' + \
                  data_db.Date.iloc[p01].strftime('%d%m%y') + '.png'
    f.savefig(os.path.join(os.getcwd(), cur_day, save_string))

    return days, zM, z28, z14, z7, z5, ok


######################################################################################################################
# Ratio plot for insufficiency defining: Death vs Recovered
def ratio_plot(data_db, base_country='Israel', w_range=[7, 14], beg_segment=110):
    data_db['Date'] = pd.to_datetime(data_db['Date'])
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for w in w_range:
        z = piecewise_linear_polyfit_by_n(data_db, w)
        ax1.plot(data_db.Date[w + 5:], z, linewidth=2)
    ax1.grid()
    ax1.set_title(base_country + ' - ' + 'Deaths to recovered ratio', fontsize=16)
    ax1.legend((str(w_range[0]) + ' days estimation', str(w_range[-1]) + ' days estimation'))

    db = data_db[beg_segment:]
    for w in w_range:
        zsegm = piecewise_linear_polyfit_by_n(db, w)
        ax2.plot(db.Date[w + 5:], zsegm, linewidth=2)
    ax2.grid()
    ax2.set_title('Deaths to recovered ratio (zoom)', fontsize=16)
    ax2.legend((str(w_range[0]) + ' days estimation', str(w_range[-1]) + ' days estimation'))
    f.autofmt_xdate()

    last_data_day = data_db.Date.max().strftime('%d%m%y')
    cur_day = time.strftime("%d%m%Y")
    save_string = last_data_day + '_D2Rratio_' + base_country + '.png'
    f.savefig(os.path.join(os.getcwd(), cur_day, save_string))


######################################################################################################################
# plot of ras data and smooth data together for concrete case
def daily_plot_raw_and_smooth(data_db, base_country='Israel', input='Deaths'):
    dates = pd.to_datetime(data_db.Date)
    in_field = 'New' + input

    f, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot(dates, data_db[in_field])
    ax1.plot(dates[:-3], smooth_vector(data_db[in_field])[:-3], linewidth=3)
    ax1.grid()

    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_xlim((dates.min(), dates.max()))
    ax1.legend(('Daily ' + input + ' Raw Data', 'Daily ' + input + ' Smoothed Data'))
    ax1.set_ylabel('Number of individuals', fontsize=16)
    ax1.set_title(base_country + ' situation - Daily ' + input + ' (data from ' + str(dates.max())[:10] + ')',
                  fontsize=16)

    last_data_day = data_db.Date.max().strftime('%d%m%y')
    save_string = last_data_day + '_daily' + input + 'Situation_' + base_country + '.png'
    cur_day = time.strftime("%d%m%Y")
    f.savefig(os.path.join(os.getcwd(), cur_day, save_string))


######################################################################################################################
# plot with twin axes
def daily_plot_twin_axes(data_db, base_country='Israel', input=['Deaths', 'Confirmed']):
    in_field1 = 'New' + input[0]
    in_field2 = 'New' + input[1]
    data1 = data_db[in_field1]
    data2 = data_db[in_field2]
    dates = pd.to_datetime(data_db['Date'])

    fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
    color = 'tab:red'
    # ax1.set_xlabel('time [month]')
    ax1.set_ylabel('Daily ' + input[0], color=color, fontsize=18, fontweight='bold')
    ax1.plot(dates[:-3], smooth_vector(data1)[:-3], color=color, linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.fmt_xdata = formatter
    fig.autofmt_xdate()
    ax1.grid(axis='x')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Daily ' + input[1], color=color, fontsize=18, fontweight='bold')
    # we already handled the x-label with ax1
    ax2.plot(dates[:-3], smooth_vector(data2)[:-3], color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title(base_country + ' - ' + 'Daily situation', fontsize=16)
    fig.legend(('Daily ' + input[0] + ' (7 days smoothed)', 'Daily ' + input[1] + ' (7 days smoothed)'), fontsize=14,
               loc=[.06, 0.70])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    last_data_day = data_db.Date.max().strftime('%d%m%y')
    cur_day = time.strftime("%d%m%Y")
    save_string = last_data_day + '_dailySituation_' + base_country + '.png'
    fig.savefig(os.path.join(os.getcwd(), cur_day, save_string))


######################################################################################################################
# save especial variables into the csv file
def save_to_csv(countries, deathVsRecovered, ConfirmedRecoveredDeath):
    cur_day = time.strftime("%d%m%Y")
    file_name = os.path.join(os.getcwd(), cur_day, 'deathVsRecovered.csv')
    with open(file_name, mode='w') as csv_file:
        fieldnames = ['country', 'Confirmed', 'Recovered', 'Death', 'DRratio', 'DRoffset', 'DRstableDays',
                      'DRratio28', 'DRratio14', 'DRratio07', 'DRratio05']

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for c in range(len(deathVsRecovered)):
            writer.writerow({'country': countries[c],
                             'Confirmed': int(ConfirmedRecoveredDeath[c, 0]),
                             'Recovered': int(ConfirmedRecoveredDeath[c, 1]),
                             'Death': int(ConfirmedRecoveredDeath[c, 2]),
                             'DRratio': round(deathVsRecovered[c, 0], 6),
                             'DRoffset': round(deathVsRecovered[c, 1], 3),
                             'DRstableDays': int(deathVsRecovered[c, 2]),
                             'DRratio28': round(deathVsRecovered[c, 3], 6),
                             'DRratio14': round(deathVsRecovered[c, 4], 6),
                             'DRratio07': round(deathVsRecovered[c, 5], 6),
                             'DRratio05': round(deathVsRecovered[c, 6], 6)})


######################################################################################################################
# Death to Recovery Analysis: special cases in Israel
def death2recovery(base_country='Israel'):
    # Israel insufficiency for various cases
    base_country_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), base_country + '_db.csv')
    base_db = pd.read_csv(base_country_file)
    data_db = base_db.copy()
    
    if data_db.Recovered.max() == -data_db.NewRecovered.min():
        # Cut the data by last valid recovered value
        threshDays = data_db[data_db.Recovered == data_db.Recovered.max()].index.values[-1] - 1
        data_db = data_db.loc[:threshDays, :]
    
    len_data = len(data_db.Confirmed)
    
    db = data_db[:70]
    # ('2020-02-21  2020-04-10')
    p00 = 0
    p01 = 49
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    
    # ('2020-04-11  2020-04-20')
    p00 = 50
    p01 = 59
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)

    db = data_db[:155]
    # ('2020-04-21  2020-07-02')
    p00 = 60
    p01 = 132
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    # ('2020-07-02  2020-07-19')
    p00 = 132
    p01 = 149
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    
    db = data_db[:265]
    # ('2020-07-26 2020-09-25')
    p00 = 156
    p01 = 217
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    print('second small => SEGER 18.9.20')
    
    p00 = 190
    p01 = 197
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    print('second unstable situation 2020-08-29 2020-09-05 =>Gmazo''s Ramzor')
    
    p00 = 201
    p01 = 206
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    print('SEGER 2020-09-18 2020-09-21')
    
    p00 = 210
    p01 = 213
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    print('second unstable situation 2020-08-29 2020-09-05 =>Gmazo''s Ramzor')
    
    # ('2020-07-26 2020-09-25')
    p00 = 156
    p01 = 244
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)
    
    p00 = 245
    p01 = 261
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)

    # February 2021
    db = data_db.copy()
    p00 = 346
    p01 = 374
    print([db.Date.iloc[p00], db.Date.iloc[p01]])
    days, zM, z28, z14, z7, z5, ok = situation_plot(db, base_country, p00, p01)


######################################################################################################################
# Begin
today = time.strftime("%d%m%Y")
full_data_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), time.strftime("%d%m%Y") + 'complete_data.csv')
world_pop_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'world_population_csv.csv')

clean_db = pd.read_csv(full_data_file)
# Remove Israel to insert later at first place in the country list
all_countries = clean_db[clean_db['Country'].str.contains('Israel') != True]
all_countries = all_countries['Country'].unique()
# Add world to the beginning
all_countries = np.insert(all_countries, 0, 'world')
# Add First of all Israel
all_countries = np.insert(all_countries, 0, 'Israel')

# Some countries
some_countries = ['Israel', 'world', 'US', 'Russia', 'Brazil', 'Italy', 'Iran', 'Spain', 'France', 'Belgium', 'Sweden',
                  'Singapore', 'Switzerland', 'Turkey', 'Denmark', 'Germany', 'Austria', 'Australia', 'South Korea',
                  'Japan', 'Portugal', 'Norway', 'Qatar', 'Iceland', 'New Zealand', 'Panama', 'Estonia', 'Cyprus']

# To calculate Insufficiency Indicator for all countries
# and generate the corresponding Figures, enable the flag do_all = True
do_all = False
some = False

if do_all:
    countries = all_countries
elif some:
    countries = some_countries
else:
    countries = ['Israel']
# also the possibility for run if already some_countries were running or
# are not relevant due to absent some data
# Caution: In 'United Kingdom', 'Netherlands' the recovered data are absent
# remove_countries = ['United Kingdom', 'Netherlands', 'Ireland']
remove_countries = ['Summer Olympics 2020']
countries = [item for item in countries if item not in remove_countries]

stdoutOrigin = sys.stdout

# Initialisation
deathVsRecovered = np.zeros((len(countries), 7))
ConfirmedRecoveredDeath = np.zeros((len(countries), 3))
ccount = 0

for base_country in countries:
    # For Example Israel
    # base_country = 'Israel'  # may be: 'world' or country name like 'Russia' from the complete_data.csv
    fout = open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'insuffiency_log.txt'), 'a')
    sys.stdout = MyWriter(sys.stdout, fout)

    print(base_country)

    base_country_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), base_country + '_db.csv')

    if os.path.exists(base_country_file):
        base_db = pd.read_csv(base_country_file)

    elif os.path.exists(full_data_file):
        clean_db = pd.read_csv(full_data_file)
        world_population = pd.read_csv(world_pop_file)
        clean_db['Date'] = pd.to_datetime(clean_db['Date'])

        # Sort by Date
        daily = clean_db.sort_values(['Date', 'Country', 'State'])
        try:
            base_db = country_analysis(clean_db, world_population, country=base_country, state='', plt=True,
                                       fromFirstConfirm=True, num_days_for_rate=60)
            if base_country[-1] == '*':
                base_country = base_country[:-1]
                base_db['Country'] = base_country
            base_db.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), base_country + '_db.csv'), index=False)
        except:
            print('No luck loading data of ' + base_country)
            continue

    data_db = base_db.copy()

    if data_db.Recovered.max() == -data_db.NewRecovered.min():
        # Cut the data by last valid recovered value
        threshDays = data_db[data_db.Recovered == data_db.Recovered.max()].index.values[-1] - 1
        data_db = data_db.loc[:threshDays, :]

    len_data = len(data_db.Confirmed)

    # Plots
    days, zM, z28, z14, z7, z5, ok = situation_plot(data_db, base_country)
    if ok:
        ratio_plot(data_db, base_country)
        daily_plot_raw_and_smooth(data_db, base_country)
        daily_plot_twin_axes(data_db, base_country)

    # Store the data per country
    deathVsRecovered[ccount, :] = [zM[0], zM[1], days, z28[0], z14[0], z7[0], z5[0]]
    ConfirmedRecoveredDeath[ccount, :] = [base_db.Confirmed.iloc[-1], base_db.Recovered.iloc[-1],
                                          base_db.Deaths.iloc[-1]]
    ccount = ccount + 1
    plt.pause(.001)
    plt.close('all')

# save to csv file
save_to_csv(countries, deathVsRecovered, ConfirmedRecoveredDeath)

######################################################################################################
# Death to Recovery Analysis: special cases in Israel
death2recovery(base_country='Israel')

# plt.show()
plt.close('all')

fout.close()
# sys.stdout.close()
sys.stdout = stdoutOrigin
