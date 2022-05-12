"""
Created on Thursday Mar 26 2020
Sveta Raboy
based on
https://www.kaggle.com/bardor/covid-19-growing-rate
https://github.com/CSSEGISandData/COVID-19
https://github.com/imdevskp
https://www.kaggle.com/yamqwe/covid-19-status-israel

"""
import sys
import extract_who_data
from Utils import *
import time

# Begin
full_data_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), time.strftime("%d%m%Y") + 'complete_data.csv')
world_pop_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'world_population_csv.csv')

# To create all the Figures on the first run, enable the flag create_all_figures = True
create_all_figures = False

if os.path.exists(full_data_file):
    clean_db = pd.read_csv(full_data_file)
    world_population = pd.read_csv(world_pop_file)
    first_plt = False
else:
    os.makedirs(os.path.join(os.getcwd(), time.strftime("%d%m%Y")), exist_ok=True)
    # Extract Data from World Health Organisation (WHO)
    clean_db, world_population = extract_who_data.extract_data()
    first_plt = create_all_figures

stdoutOrigin = sys.stdout
fout = open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'world_status_log.txt'), 'a')
sys.stdout = MyWriter(sys.stdout, fout)

clean_db['Date'] = pd.to_datetime(clean_db['Date'])

# Sort by Date
daily = clean_db.sort_values(['Date', 'Country', 'State'])

# the latest day
current_date = daily.Date.max()
previous_date = daily.Date.sort_values().unique()[-2]
latest = clean_db[clean_db.Date == current_date]
previous = clean_db[clean_db.Date == previous_date]

# Date Status
# Current World Situation
inputs = ['Confirmed', 'Deaths', 'Recovered', 'Active']
dbd = daily.groupby("Date")[inputs].sum().reset_index()
dbd = growth_func(dbd, inputs, numDays=1, name='New', normalise=False)
dbd = growth_func(dbd, inputs, numDays=1, name='Growth', normalise=True)
dbd = normalise_func(dbd, inputs=['Deaths', 'Recovered', 'Active'], name='NormConfirm', normaliseTo='Confirmed',
                     factor=1, toRound=True)
if first_plt:
    fdbd1 = scatter_country_plot(dbd)
    fdbd2 = scatter_country_plot(dbd, prefix='New', fname=' Daily New Cases')
    fdbd3 = scatter_country_plot(dbd, prefix='Growth',  fname=' Growing rate in % a day')
    fdbd4 = scatter_country_plot(dbd, inputs=['Deaths', 'Recovered', 'Active'], prefix='NormConfirm',
                                 factor=100.0, fname=' Normalised for Total Confirmed Cases - '
                                                     'Probability to Case If infected by the virus (%)')
    fdbd5 = scatter_country_plot(dbd, inputs=['Deaths'], base='Recovered', add_growth_rates=True,
                                 fname=' Cases Ratio: Deaths vs Recovered')
    with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), current_date.strftime('%d%m%y')
              + '_World_Various_Cases .html'), 'a') as f:
        f.write(fdbd1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fdbd2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fdbd3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fdbd4.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fdbd5.to_html(full_html=False, include_plotlyjs='cdn'))

current_date_countries = latest.groupby("Country")[inputs].sum().reset_index()
previous_date_countries = previous.groupby("Country")[inputs].sum().reset_index()
growth_date = (current_date_countries[inputs] - previous_date_countries[inputs]).astype(int)  # .clip(0)
growth_date.loc[:, 'Country'] = current_date_countries['Country'].values
latest_day_status = pd.DataFrame(columns=inputs)
latest_day_status.loc[0] = current_date_countries[inputs].sum().astype(int)
cnt = 1
for k in inputs:
    max_cur = current_date_countries[current_date_countries[k] == current_date_countries[k].max()]
    growth_cur = growth_date[growth_date[k] == growth_date[k].max()]
    latest_day_status.loc[1, k] = max_cur[k].values[0].astype(int)
    latest_day_status.loc[2, k] = max_cur['Country'].values[0]
    latest_day_status.loc[3, k] = growth_cur[k].values[0].astype(int)
    latest_day_status.loc[4, k] = growth_cur['Country'].values[0]
latest_day_status.index = ['Total', 'Current Max', 'Country Current Max', 'Current Max Growth ', 'Country Max Growth']
print('Day Status ' + current_date.strftime('%d/%m/%y') + ':')
print(latest_day_status)

# Create Latest World Summary Table
header_columns = ['Current Day', 'Total', 'Max Value', 'Country of Max Value', 'Max Growth ', 'Country of Max Growth']
table_title = 'CoViD-19 World Daily Situation Summarise for ' + str(current_date_countries.shape[0]) + ' countries'
if first_plt:
    create_table(latest_day_status, current_date, inputs, header_columns, table_title, height='50%')

for k in inputs:
    growth_date['New' + k] = growth_date[k]
    current_date_countries = pd.concat([current_date_countries, growth_date['New' + k]], axis=1, sort=False)
    del growth_date[k]

current_date_countries, pop_latest = add_pop_age_data(current_date_countries, world_population)
dbd.loc[:, 'Population'] = pop_latest.Population.sum()
dbd.loc[:, 'Age'] = pop_latest.Age.median()
dbd.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'world_db.csv'), index=False)

for k in inputs:
    current_date_countries[k + ' / 1M pop'] = (current_date_countries[k] * 1e6 /
                                               current_date_countries['Population']).fillna(0).clip(0).astype(int)

# World Daily Situation in Bars for countries since 1m population
onem_current_date_countries = current_date_countries[current_date_countries['Population'] > 1e6]
if first_plt:
    countries_bar(onem_current_date_countries, current_date, count=35,
                  fname='_Most_Values_Daily_Situation_For_Countries_Since_1M_Population')

# Add Population & Age information to the Data
current_date_countries = current_date_countries.append(current_date_countries.sum(numeric_only=True).rename('Total'),
                                                       ignore_index=True)
current_date_countries.Country = current_date_countries.Country.fillna('Total')
srt_current = current_date_countries.sort_values('Confirmed', ascending=0)
header_columns = srt_current.Country.to_list()
header_columns.insert(0, 'Current Day')
inputs = srt_current.keys()[1:].values
table_title = 'CoViD-19 World Daily Situation for ' + str(srt_current.shape[0]-1) + ' countries'
# Create Latest World Status Table
if first_plt:
    create_table(srt_current, current_date, inputs, header_columns, table_title, height='300%')

# Create map for latest day
if first_plt:
    create_map(latest, world_population)

# bars for Cases for all countries
create_bars = first_plt
if create_bars:
    case_groupby_bar(daily, world_population, groupby=['Date', 'State', 'Country'],
                     inputs=['Confirmed', 'Recovered', 'Deaths', 'Active'],
                     threshould=[int(latest_day_status.Confirmed[1]*0.75), int(latest_day_status.Recovered[1]*0.75),
                                 int(latest_day_status.Deaths[1]*0.75), int(latest_day_status.Active[1]*0.75)])
#############################################################################


# Israel
base_country = 'Israel'
israel_events_file = (os.path.join(os.getcwd(), 'israelWhatWasWasEng.xlsx'))
if os.path.exists(israel_events_file):
    Events = pd.read_excel(israel_events_file, sheet_name='israelWhatWasWas')
else:
    dates = pd.to_datetime(['2020/03/10', '2020/03/15', '2020/03/18', '2020/04/26', '2020/04/28', '2020/04/29',
                            '2020/05/03', '2020/05/07', '2020/05/27', '2020/05/28'], format='%Y/%m/%d')
    dates = dates.append(pd.date_range('2020/04/08', '2020/04/15'))
    event = ['Purim', 'SchoolClosed', 'LockdownBegins', 'LockdownEnds', 'MemorialDay', 'Independence',
             'SchoolOpen', 'MarketOpen', 'PubsOpen', 'Shavuot'] + ['Pesah' for i in range(8)]
    Events = pd.DataFrame({'Date': dates, 'Event': event})

israel_db = country_analysis(clean_db, world_population, country=base_country, state='', plt=first_plt,
                             fromFirstConfirm=True, events=Events)
israel_db.to_csv(os.path.join(os.getcwd(), base_country + '_db.csv'), index=False)
################################################################################################

# Israel and some countries
# World Daily Situation in Bars for countries since 1m population
countries_cases = first_plt
if countries_cases:

    all_countries = daily['Country'].unique()
    # remove Liechtenstein which is without update
    current_date_countries = current_date_countries[current_date_countries['Country'].str.contains('Liechtenstein') != True]

    dth = current_date_countries[current_date_countries['Deaths / 1M pop'] > 0.75 * israel_db['NormPopDeaths'].max()]
    dth_countries = dth[dth['Deaths / 1M pop'] < 1.05 * israel_db['NormPopDeaths'].max()]['Country'].unique()
    pop = current_date_countries[current_date_countries.Population > 0.75 * israel_db.Population.max()]
    pop_countries = pop[pop.Population < 1.05 * israel_db.Population.max()]['Country'].unique()
    cnf = current_date_countries[current_date_countries['Confirmed / 1M pop'] > 0.85 * israel_db['NormPopConfirmed'].max()]
    cnf_countries = cnf[cnf['Confirmed / 1M pop'] < 1.15 * israel_db['NormPopConfirmed'].max()]['Country'].unique()

    countries1 = ['Italy', 'Iran', 'Spain', 'France', 'US', 'United Kingdom', 'Russia', 'Brazil']
    # 'Netherlands', 'Belgium', 'Portugal', 'Norway', 'Iceland', 'Ireland']
    countries2 = ['Taiwan*', 'New Zealand', 'Japan', 'South Korea', 'Singapore', 'Switzerland', 'Turkey']
    # 'Sweden', 'Denmark', 'Germany', 'Austria']

    ddb = None
    cnfdb = None
    pdb = None
    cdb1 = israel_db
    cdb2 = israel_db

    countries1 = [item for item in countries1 if item not in cnf_countries]
    countries1 = [item for item in countries1 if item not in pop_countries]
    countries1 = [item for item in countries1 if item not in dth_countries]
    countries2 = [item for item in countries2 if item not in cnf_countries]
    countries2 = [item for item in countries2 if item not in pop_countries]
    countries2 = [item for item in countries2 if item not in dth_countries]

    cnf_countries = [item for item in cnf_countries if item not in pop_countries]
    cnf_countries.append(israel_db.Country.values[0])
    for country in dth_countries:
        curr = country_analysis(clean_db, world_population, country=country, state=None)
        ddb = pd.concat([ddb, curr], axis=0, sort=False, ignore_index=True)
    for country in cnf_countries:
        curr = country_analysis(clean_db, world_population, country=country, state=None)
        cnfdb = pd.concat([cnfdb, curr], axis=0, sort=False, ignore_index=True)
    for country in pop_countries:
        curr = country_analysis(clean_db, world_population, country=country, state=None)
        pdb = pd.concat([pdb, curr], axis=0, sort=False, ignore_index=True)
    for country in countries1:
        curr = country_analysis(clean_db, world_population, country=country, state=None)
        cdb1 = pd.concat([cdb1, curr], axis=0, sort=False, ignore_index=True)
    for country in countries2:
        curr = country_analysis(clean_db, world_population, country=country, state=None)
        cdb2 = pd.concat([cdb2, curr], axis=0, sort=False, ignore_index=True)

    create_plt = first_plt
    # Countries Cases Plot
    if create_plt:
        # Days
        threshDays = [1, 1]
        # Value ( Default)
        threshValues = [1, 1]
        for cnt in range(2):
            inputs_kind = cnt
            # Countries Cases Plot from threshold = Number_of_Input
            if inputs_kind:
                inputs = ['Confirmed', 'Deaths']
            else:
                inputs = ['Active', 'Recovered']
            prefixes = ['NormPop', 'New', 'NormConfirm']
            factors = [1, 1, 1]
            add_growth_rate = [False, False, False]
            logs = [False, False, False]
            for cprx in range(len(prefixes)):
                cases = inputs
                prefix = prefixes[cprx]
                factor = factors[cprx]
                add_growth_rates = add_growth_rate[cprx]
                log = logs[cprx]
                if cprx == 2:
                    threshValues = [0, 0]
                    if inputs_kind:
                        cases = ['Active', 'Deaths']
                with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), current_date.strftime('%d%m%y')
                          + '_Days_since_the_' + str(threshDays[0]) + 'th_from_' + str(threshValues[0]) + 'th_' + prefix
                          + '_for_' + inputs[1] + '_Cases_For_Various_Countries.html'), 'a') as f:
                    fsc1 = case_thresh_plot(cdb1, threshDays=threshDays, inputs=cases, prefix=prefix, factor=factor,
                                            fname='Countries vs Israel', log=log, threshValues=threshValues,
                                            add_growth_rates=add_growth_rates)
                    fsc2 = case_thresh_plot(cdb2, threshDays=threshDays, inputs=cases, prefix=prefix, factor=factor,
                                            fname='Others Countries vs Israel', log=log,  threshValues=threshValues,
                                            add_growth_rates=add_growth_rates)
                    fsc3 = case_thresh_plot(cnfdb, threshDays=threshDays, inputs=cases, prefix=prefix, factor=factor,
                                            fname='Countries with the -15%:15% Confirmed Case Values as in Israel',
                                            threshValues=threshValues, add_growth_rates=add_growth_rates)
                    fsc4 = case_thresh_plot(ddb, threshDays=threshDays, inputs=cases, prefix=prefix, factor=factor, log=log,
                                            fname='Countries with the -25%:5% Deaths Case Values as in Israel',
                                            threshValues=threshValues, add_growth_rates=add_growth_rates)
                    fsc5 = case_thresh_plot(pdb, threshDays=threshDays, inputs=cases, prefix=prefix, factor=factor, log=log,
                                            fname='Countries with the -25%:5% Population as in Israel',
                                            threshValues=threshValues, add_growth_rates=add_growth_rates)
                    f.write(fsc5.to_html(full_html=False, include_plotlyjs='cdn'))
                    f.write(fsc4.to_html(full_html=False, include_plotlyjs='cdn'))
                    f.write(fsc3.to_html(full_html=False, include_plotlyjs='cdn'))
                    f.write(fsc2.to_html(full_html=False, include_plotlyjs='cdn'))
                    f.write(fsc1.to_html(full_html=False, include_plotlyjs='cdn'))

############################################################

choisen_cases = first_plt
if choisen_cases:
    cdb = None
    countries = ['Israel', 'Australia']
    for country in countries:
        curr = country_analysis(clean_db, world_population, country=country, state=None)
        cdb = pd.concat([cdb, curr], axis=0, sort=False, ignore_index=True)

    # Days
    threshDays = [1, 1]
    # Value ( Default)
    threshValues = [1, 1]
    # Countries Cases Plot
    with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), current_date.strftime('%d%m%y') + '_Days_since_the_'
              + str(threshDays[0]) + 'th_from_' + str(threshValues[0]) + 'th_' + '_for_' + 'All_Cases_For_'
              + str(len(countries)) + '_Countries.html'), 'a') as f:

        for cnt in range(2):
            inputs_kind = cnt
            # Countries Cases Plot from threshold = Number_of_Input
            if inputs_kind:
                inputs = ['Confirmed', 'Deaths']
            else:
                inputs = ['Active', 'Recovered']
            prefixes = ['NormPop', 'New', 'NormConfirm']
            factors = [1, 1, 1]
            add_growth_rate = [False, False, False]
            logs = [False, False, False]
            for cprx in range(len(prefixes)):
                cases = inputs
                prefix = prefixes[cprx]
                factor = factors[cprx]
                add_growth_rates = add_growth_rate[cprx]
                log = logs[cprx]
                if cprx == 2:
                    threshValues = [0, 0]
                    if inputs_kind:
                        cases = ['Active', 'Deaths']
                fsc = case_thresh_plot(cdb, threshDays=threshDays, inputs=cases, prefix=prefix, factor=factor,
                                       fname='Countries', log=log, threshValues=threshValues,
                                       add_growth_rates=add_growth_rates)
                f.write(fsc.to_html(full_html=False, include_plotlyjs='cdn'))

fout.close()
# sys.stdout.close()
sys.stdout = stdoutOrigin
