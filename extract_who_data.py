"""
Created on Thursday Mar 26 2020
Sveta Raboy
based on
https://github.com/CSSEGISandData/COVID-19
https://github.com/imdevskp
https://www.kaggle.com/yamqwe/covid-19-status-israel
"""

import pandas as pd
import wget
import time
import os
import numpy as np
import datetime


# function to change value
def change_val(date, ref_col, val_col, dtnry, full_table):
    for key, val in dtnry.items():
        full_table.loc[(full_table['Date'] == date) & (full_table[ref_col] == key), val_col] = val
    return full_table


def add_values_according_lat(input_data, cng_data):
    lats = input_data['Lat'].round(4)
    out_data = None
    for lat in lats:
        cur_in_data = input_data[input_data['Lat'].round(4) == lat].reset_index()
        longs = cur_in_data['Long'].round(4)
        cur_add_data = cng_data[cng_data['Lat'].round(4) == lat].reset_index()
        if (cur_in_data['Lat'].round(4).values == cur_add_data['Lat'].round(4).values).all():
            for long in longs:
                state = cur_in_data[cur_in_data['Long'].round(4) == long]['State']
                cur_add_data = cur_add_data[cur_add_data['Long'].round(4) == long]
                if not cur_add_data[cur_add_data['State'] == state].values.any():
                    cur_add_data['State'] = cur_add_data['Country']
                if out_data is None:
                    out_data = cur_add_data
                else:
                    out_data = pd.concat([out_data, cur_add_data], axis=0, sort=False)
        else:
            cur_add_data = cur_in_data
            out_data = pd.concat([out_data, cur_add_data[['State', 'Country', 'Lat', 'Long']]], axis=0, sort=False)

    return out_data


def extract_data(filename=None, world_pop_file=(os.path.join(os.getcwd(), 'world_population.xlsx'))):

    # if filename not exist
    if not filename:
        # Data Download
        urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']

        # download data
        filename = []
        for url in urls:
            cur_file = wget.download(url)
            new_name = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), time.strftime("%d%m%Y")+cur_file)
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(cur_file, new_name)
            filename.append(new_name)

    # DataSet
    db_can = []
    db = []
    country_num = []
    state_num = []
    for k in range(len(filename)):
        cur_file = pd.read_csv(filename[k])
        cur_file = cur_file.rename(columns={"Country/Region": "Country", "Province/State": "State"})
        cur_file = cur_file.sort_values(['Country', 'State'])
        # Cleaning Data
        # removing canada's recovered values
        cur_file = cur_file[cur_file['State'].str.contains('Recovered') != True]
        # removing Repatriated Travellers's values in Canada
        cur_file = cur_file[cur_file['State'].str.contains('Repatriated Travellers') != True]
        # removing Diamond Princess's values in Canada
        cur_file = cur_file[cur_file['State'].str.contains('Diamond Princess') != True]
        # removing Unknown values in State
        cur_file = cur_file[cur_file['State'].str.contains('Unknown') != True]
        # removing Diamond Princess ship
        cur_file = cur_file[cur_file['Country'].str.contains('Diamond Princess') != True]
        # removing  MS Zaandam  ship
        cur_file = cur_file[cur_file['Country'].str.contains('MS Zaandam') != True]
        # removing county wise data to avoid double counting
        cur_file = cur_file[cur_file['State'].str.contains(',') != True]
        # renaming countries, regions, provinces
        cur_file['Country'] = cur_file['Country'].replace('Korea, South', 'South Korea')
        cur_file['Country'] = cur_file['Country'].replace('West Bank and Gaza', 'Palestinian Authority')
        cur_file.loc[cur_file['State'].isna() != False, 'State'] = cur_file.loc[cur_file['State'].isna() != False, 'Country']
        # summarise data for Canada (not exists)
        can = cur_file[cur_file['Country'] == 'Canada'].reset_index()
        # removing Canada's values: not consistent in these 3 files
        cur_file = cur_file[cur_file['Country'].str.contains('Canada') != True]
        cur_file = cur_file.reset_index()
        country_num.append(cur_file['Country'].unique().shape)
        state_num.append(cur_file['State'].unique().shape)
        if k > 0:
            if country_num[k] != country_num[k-1]:
                raise ValueError("Number of countries is not equal in the downloaded files. "
                                 "It will cause to wrong statistics.")
            if state_num[k] != state_num[k-1]:
                raise ValueError("Number of states is not equal in the downloaded files. "
                                 "It will cause to wrong statistics.")
        db.append(cur_file)

        if k > 1 and db_can[0]['State'].unique().shape != can['State'].unique().shape:
            temp = db_can[0].copy()
            state_max_value = temp[temp[temp.columns[-1]] == temp[temp.columns[-1]].max()]['State']
            temp.iloc[:, 4:] = 0
            temp.iloc[temp['State'] == state_max_value.values[0], 4:] = can[can.columns[4:]].values
            can = temp
        db_can.append(can)

    if os.path.exists(world_pop_file):
        world_pop = pd.read_excel(world_pop_file, sheet_name=None)
        # world_pop['Sheet1'].to_csv(os.path.join(os.getcwd(), 'world_population_csv.csv'), index=False)
        world_population_tab = world_pop['Sheet1'].sort_values('Country (or dependency)').reset_index()
        world_population_tab = pd.concat([world_population_tab['Country (or dependency)'],
                                          world_population_tab['Population 2020'],
                                         world_population_tab['Med. Age']], axis=1, sort=False)
        world_population_tab = world_population_tab.rename(
            columns={"Country (or dependency)": "Country", "Population 2020": "Population", "Med. Age": "Age"})
        world_population_tab.loc[world_population_tab['Age'].str.contains('N.A.') == True, 'Age'] = np.nan
        world_population_tab.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'world_population_csv.csv'), index=False)

    confirm = pd.concat([db[0], db_can[0]], ignore_index=True)
    deaths = pd.concat([db[1], db_can[1]], ignore_index=True)
    recovery = pd.concat([db[2], db_can[2]], ignore_index=True)

    # Merge Data
    # First 4 columns are ['Index', 'State', 'Country', 'Lat', 'Long']
    dates = confirm.columns[5:]
    # Unpivot a DataFrame from wide to long format
    conf_long = confirm.melt(id_vars=['State', 'Country', 'Lat', 'Long'], value_vars=dates, var_name='Date', value_name='Confirmed')
    deaths_long = deaths.melt(id_vars=['State', 'Country', 'Lat', 'Long'], value_vars=dates, var_name='Date', value_name='Deaths')
    recv_long = recovery.melt(id_vars=['State', 'Country', 'Lat', 'Long'], value_vars=dates, var_name='Date', value_name='Recovered')
    full_table = pd.concat([conf_long, deaths_long['Deaths'], recv_long['Recovered']], axis=1, sort=False)

    # Fixing off data from WHO data
    # new values
    feb_12_conf = {'Hubei': 34874}

    # changing values
    full_table = change_val('2/12/20', 'State', 'Confirmed', feb_12_conf, full_table)

    # checking values
    full_table[(full_table['Date'] == '2/12/20') & (full_table['State'] == 'Hubei')]

    # Create new variables with calculation based on the exist variables
    full_table['Active'] = (full_table['Confirmed'] - full_table['Recovered'] - full_table['Deaths']).clip(0).astype(int)

    # Saving full data
    full_table.to_csv(os.path.join(os.getcwd(),  time.strftime("%d%m%Y"),
                                   time.strftime("%d%m%Y") + 'complete_data.csv'), index=False)
    full_table.to_csv(os.path.join(os.getcwd(),  time.strftime("%d%m%Y"), 'complete_data.csv'), index=False)

    return full_table, world_population_tab
