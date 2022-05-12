"""
Created on Jan 09 2021
Sveta Raboy and Doron Bar
database analysis from
https://data.gov.il/dataset/covid-19

Israel cities coordinates data
https://data-israeldata.opendata.arcgis.com/
"""

import json
import requests
import sys
import extract_israel_data
from Utils import *
import time
import pandas as pd
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import datetime
import numpy as np
import warnings

plt.style.use('default')
warnings.filterwarnings("ignore")

line_statistic_plot_log = None
line_statistic_plot_logYes = True
line_statistic_plot_fix_date = False


# Number with comma for every 3 digits in string
def num2strWithComma(num1):
    s1 = ''
    while num1 > 0:
        s1 = str(num1 % 1000) + ',' + s1
        num1 = num1//1000
    return s1[:-1]


# data line plot
def line_statistic_plot(db, base, fields, title, ylabel, legend, text, save_name, log=None, fix_date=False):
    f, ax = plt.subplots(figsize=(18, 6))
    date = db[base]
    date = pd.to_datetime(date)
    len_data = len(date)
    colors = plotly.colors.qualitative.Dark24  # ['blue', 'green', 'magenta', 'black', 'red', 'cyan', 'yellow']
    sum_case = []
    for cnt in range(len(fields)):
        case = fields[cnt]
        sum_case.append(db[case].max())
        plt.plot(date, db[case], zorder=1, color=colors[cnt], linewidth=3)

    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(legend, fontsize=14)

    if fix_date:
        datemin = pd.to_datetime('2020-03-01')
        datemax = pd.to_datetime('2021-05-01')
    else:
        datemin = date.min()
        datemax = date.max()

    ax.set_xlim(datemin, datemax)
    ax.grid(True)
    # rotate and align the tick labels so they look better
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.fmt_xdata = formatter
    f.autofmt_xdate()

    if log:
        ax.set_yscale('log')
    if text is not None:
        tline = 0.25*max(sum_case)
        for kk in range(len(text)):
            plt.plot((text[kk], text[kk]), (0, tline), '-k', linewidth=3)
            plt.text(text[kk], 1.1*tline, text[kk].strftime('%d/%m/%y'), horizontalalignment='center',
                     fontweight='bold', fontsize=14)

    save_string = save_name + datemax.strftime('%d%m%y') + '.png'
    f.savefig(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), save_string))


###################################################################################################################
# Begin
full_data_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), time.strftime("%d%m%Y") + '_loaded_files.csv')

if os.path.exists(full_data_file):
    files_db = pd.read_csv(full_data_file, encoding="ISO-8859-8")
    first_plt = False
else:
    os.makedirs(os.path.join(os.getcwd(), time.strftime("%d%m%Y")), exist_ok=True)
    # Extract Data from Israel Dataset COVID-19
    files_db = extract_israel_data.extract_israel_data()
    first_plt = True

# Print LOG to file
stdoutOrigin = sys.stdout
fout = open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'israel_status_log.txt'), 'a', encoding="ISO-8859-8")
sys.stdout = MyWriter(sys.stdout, fout)

text = None
# text = pd.date_range('2020-04-01', '2021-04-01', freq="MS")

###################################################################################################################
# Isolation
isolated = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('isolation').values.argmax()])
id = files_db.current_file_name.str.find('isolation').values.argmax()
print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])
base = 'date'
isolated[base] = pd.to_datetime(isolated[base])
isolated = isolated.sort_values([base])
for key in isolated.keys():
    try:
        isolated.loc[isolated[key].str.contains('15>') == True, key] = 15
        isolated[key] = isolated[key].astype(int)
    except:
        pass

iso1 = isolated.new_contact_with_confirmed.astype(int).sum()
iso2 = isolated.new_from_abroad.astype(int).sum()

title = 'Israel (data from ' + isolated[base].max().strftime('%d/%m/%y') + ') - isolated persons, total ' \
        + num2strWithComma(iso1+iso2) + ', now ' + str(isolated.isolated_today_contact_with_confirmed.iloc[-1]
        + isolated.isolated_today_abroad.iloc[-1])
ylabel = 'Number of individuals'
legend = ('Isolated due to contact with confirmed, total ' + num2strWithComma(iso1),
          'Isolated due to arrived from abroad, total ' + num2strWithComma(iso2))
save_name = 'israelIsolatedPersons_'
fields = ['isolated_today_contact_with_confirmed', 'isolated_today_abroad']

# plot Isolated Total
line_statistic_plot(isolated, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

# plot isolated daily
fields = ['new_contact_with_confirmed', 'new_from_abroad']
save_name = 'israelIsolatedPersons_Daily_'
title = 'Israel (data from ' + isolated[base].max().strftime('%d/%m/%y') + ') - Daily isolated persons, total ' \
        + str(iso1+iso2) + ', now ' + str(isolated.isolated_today_contact_with_confirmed.iloc[-1]
        + isolated.isolated_today_abroad.iloc[-1])
line_statistic_plot(isolated, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)
del isolated


###################################################################################################################
# Medical Staff
coronaMediaclStaffD = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('medical_staff').values.argmax()])
id = files_db.current_file_name.str.find('medical_staff').values.argmax()
print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])

base = 'Date'
coronaMediaclStaffD[base] = pd.to_datetime(coronaMediaclStaffD[base])
coronaMediaclStaffD = coronaMediaclStaffD.sort_values([base])
for key in coronaMediaclStaffD.keys():
    try:
        coronaMediaclStaffD.loc[coronaMediaclStaffD[key].str.contains('<15') == True, key] = 15
        coronaMediaclStaffD[key] = coronaMediaclStaffD[key].astype(int)
    except:
        pass

ylabel = 'Number of individuals'
title = 'Israel - medical staff confirmed (data from ' + coronaMediaclStaffD[base].max().strftime('%d/%m/%y') + ')'
save_name = 'coronaMediaclStaffConfirmed_'
fields = ['confirmed_cases_physicians', 'confirmed_cases_nurses', 'confirmed_cases_other_healthcare_workers']
legend = ['Confirmed physicians', 'Confirmed nurses', 'Other confirmed healthcare staff']
# plot coronaMediaclStaffConfirmed Total
line_statistic_plot(coronaMediaclStaffD, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

# plot coronaMediaclStaffIsolated daily
title = 'Israel - medical staff in isolation (data from ' + coronaMediaclStaffD[base].max().strftime('%d/%m/%y') + ')'
fields = ['isolated_physicians', 'isolated_nurses', 'isolated_other_healthcare_workers']
legend = ['Isolated physicians', 'Isolated nurses', 'Other isolated healthcare staff']
save_name = 'coronaMediaclStaffIsolated_'
line_statistic_plot(coronaMediaclStaffD, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)
del coronaMediaclStaffD


#####################################################################################################################
# Hospitalization
try:
    hospitalization = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('hospitalization').values.argmax()])
except:
    hospitalization = pd.read_excel(files_db.current_file_path[files_db.current_file_name.str.find('hospitalization').values.argmax()])

id = files_db.current_file_name.str.find('hospitalization').values.argmax()
print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])
base = 'תאריך'
hospitalization[base] = pd.to_datetime(hospitalization[base])
hospitalization = hospitalization.sort_values([base])
for key in hospitalization.keys():
    try:
        hospitalization.loc[hospitalization[key].str.contains('15>') == True, key] = 15
        hospitalization.loc[hospitalization[key].str.contains('<15') == True, key] = 15
        hospitalization[key] = hospitalization[key].astype(int)
    except:
        pass

ylabel = 'Number of individuals'
title = 'Israel - Critical conditions (data from ' + hospitalization[base].max().strftime('%d/%m/%y') + ')'
save_name = 'israelHospitalized_'
fields = ['מונשמים', 'חולים קשה', 'מאושפזים']
legend = ('Ventilated patients', 'Seriously ill', 'Hospitalized')
# plot israelHospitalized Total
line_statistic_plot(hospitalization, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

title = 'Israel - Critical conditions mean Age division (data from ' + hospitalization[base].max().strftime('%d/%m/%y') + ')'
save_name = 'israelHospitalizedInAge_'
fields = ['גיל ממוצע מונשמים', 'גיל ממוצע חולים קשה', 'גיל ממוצע מאושפזים']
legend = ('Ventilated patients', 'Seriously ill', 'Hospitalized')
# plot israelHospitalizeInAgeTotal
line_statistic_plot(hospitalization, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

title = 'Israel - Critical conditions percentage of Women (data from ' + hospitalization[base].max().strftime('%d/%m/%y') + ')'
save_name = 'israelHospitalizedInWomens_'
fields = ['אחוז נשים מונשמות', 'אחוז נשים חולות קשה', 'אחוז נשים מאושפזות']
legend = ('Ventilated patients', 'Seriously ill', 'Hospitalized')
# plot israelHospitalizeInAgeTotal
line_statistic_plot(hospitalization, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

# plot israel Ill
title = 'Israel - ill conditions (data from ' + hospitalization[base].max().strftime('%d/%m/%y') + ')'
fields = ['חולים קל', 'חולים בינוני', 'חולים קשה']
legend = ('Light ill', 'Mild ill', 'Seriously ill')
save_name = 'illConditions_'
line_statistic_plot(hospitalization, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

# plot israel mean Age Ill
title = 'Israel - ill conditions mean Age division (data from ' + hospitalization[base].max().strftime('%d/%m/%y') + ')'
fields = ['גיל ממוצע חולים קל', 'גיל ממוצע חולים בינוני', 'גיל ממוצע חולים קשה']
legend = ('Light ill', 'Mild ill', 'Seriously ill')
save_name = 'illConditionsInAge_'
line_statistic_plot(hospitalization, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

# plot israel Women Percentage Ill
title = 'Israel - ill conditions percentage of Women (data from ' + hospitalization[base].max().strftime('%d/%m/%y') + ')'
fields = ['אחוז נשים חולות קל', 'אחוז נשים חולות בינוני', 'אחוז נשים חולות קשה']
legend = ('Light ill', 'Middle ill', 'Seriously ill')
save_name = 'illConditionsInWomens_'
line_statistic_plot(hospitalization, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)
del hospitalization

###################################################################################################################
# Recovered
try:
    recovered = pd.read_excel(files_db.current_file_path[files_db.current_file_name.str.find('recovered').values.argmax()])  # , encoding="ISO-8859-8")
except:
    recovered = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('recovered').values.argmax()])

try:
    id = files_db.current_file_name.str.find('recovered').values.argmax()
    print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])

    recoveredMeanTime = recovered.days_between_pos_and_recovery.mean()
    recoveredMedianTime = recovered.days_between_pos_and_recovery.median()
    print('Recovered Mean Time: ' + str(int(recoveredMeanTime*100)/100) + ' days')
    print('Recovered Median Time: ' + str(int(recoveredMedianTime*100)/100) + ' days')
    NN = int(recovered.days_between_pos_and_recovery.max())
    hh = np.histogram(recovered.days_between_pos_and_recovery, bins=np.arange(NN+1))
    f, ax = plt.subplots(figsize=(15, 6))
    plt.plot(hh[1][1:], hh[0], linewidth=3)
    # ax.set_yscale('log')
    plt.plot([recoveredMedianTime, recoveredMedianTime], [0, hh[0].max()], 'k--')
    plt.text(recoveredMedianTime, hh[0].max(), ' Recovered Median Time: ' + str(int(recoveredMedianTime*100)/100) + ' days')

    plt.plot([recoveredMeanTime, recoveredMeanTime], [0, hh[0][int(recoveredMeanTime)]], 'k--')
    plt.text(recoveredMeanTime, hh[0][int(recoveredMeanTime)], ' Recovered Mean Time: '
             + str(int(recoveredMeanTime*100)/100) + ' days')
    plt.grid()
    plt.xlabel('Time to recovered [days]', fontsize=16)
    plt.ylabel('Number of individuals', fontsize=16)
    try:
        data_from = pd.to_datetime(str(files_db.last_update[id]))
        plt.title('Israel - Time to recovered. Num of persons ' + str(int(hh[0].sum())) + ' (data from '
                  + data_from.strftime('%d/%m/%y') + ')', fontsize=16)
    except:
        plt.title('Israel - Time to recovered. Num of persons ' + str(int(hh[0].sum())) + ' (data from '
                  + str(files_db.last_update[id]) + ')', fontsize=16)
    save_string = 'israelRecovered_' + str(files_db.last_update[id]) + '.png'
    f.savefig(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), save_string))

except:
    print('========= !!! PROBLEM with recovered data !!! =========')
    print('    calculation of the recovered time failed')

del recovered

###################################################################################################################
# Deceased
deceased = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('deceased').values.argmax()], encoding='latin-1')

id = files_db.current_file_name.str.find('deceased').values.argmax()
print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])

deceasedMeanTime = deceased.Time_between_positive_and_death.mean()
deceasedMedianTime = deceased.Time_between_positive_and_death.median()
print('Deceased Mean Time: ' + str(int(deceasedMeanTime*100)/100) + ' days')
print('Deceased Median Time: ' + str(int(deceasedMedianTime*100)/100) + ' days')
NN = int(deceased.Time_between_positive_and_death.max())
hh = np.histogram(deceased.Time_between_positive_and_death, bins=np.arange(NN+1))
f, ax = plt.subplots(figsize=(15, 6))
plt.plot(hh[1][1:], hh[0], linewidth=3)

plt.plot([deceasedMedianTime, deceasedMedianTime], [0, hh[0].max()], 'k--')
plt.text(deceasedMedianTime, hh[0].max(), ' Deceased Median Time: ' + str(int(deceasedMedianTime*100)/100) + ' days')

plt.plot([deceasedMeanTime, deceasedMeanTime], [0, hh[0][int(deceasedMeanTime)]], 'k--')
plt.text(deceasedMeanTime, hh[0][int(deceasedMeanTime)], ' Deceased Mean Time: ' + str(int(deceasedMeanTime*100)/100) + ' days')
plt.grid()
plt.xlabel('Time to deceased [days]', fontsize=16)
plt.ylabel('Number of individuals', fontsize=16)
try:
    plt.title('Israel - Time to deceased. Num of persons ' + str(int(hh[0].sum())) + '. Num of Ventilated ' +
          str(int(deceased.Ventilated.sum())) + ' (data from ' + data_from.strftime('%d/%m/%y') + ')', fontsize=16)
except:
    plt.title('Israel - Time to deceased. Num of persons ' + str(int(hh[0].sum())) + '. Num of Ventilated ' +
          str(int(deceased.Ventilated.sum())) + ' (data from ' + str(files_db.last_update[id]) + ')', fontsize=16)    
save_string = 'israelDeceased_' + str(files_db.last_update[id]) + '.png'
f.savefig(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), save_string))
del deceased

plt.close('all')

###################################################################################################################
# Lab Test
lab_tests = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('lab_tests').values.argmax()])

id = files_db.current_file_name.str.find('lab_tests').values.argmax()
print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])
base = 'result_date'
# lab_tests.loc[lab_tests['result_date'].isna() != False, 'result_date'] = lab_tests.loc[lab_tests['result_date'].isna() != False, 'test_date']
lab_tests = lab_tests[lab_tests['result_date'].isna() != True]
N = len(lab_tests.corona_result)
lab_tests[base] = pd.to_datetime(lab_tests[base])
lab_tests = lab_tests.sort_values([base])
possible_results = lab_tests.corona_result.unique()

FirstTest = lab_tests.loc[lab_tests['is_first_Test'].str.contains('Yes') != False, ['result_date', 'corona_result']].reset_index()
first_grouped = FirstTest.groupby(['result_date', 'corona_result'],  as_index=False).count()
first = first_grouped.set_index(['result_date', 'corona_result']).unstack().fillna(0).astype(int).add_prefix('ראשון ')
del FirstTest, first_grouped
first_positive = first.xs("ראשון חיובי", level="corona_result", axis=1).values.squeeze()
first_negative = first.xs("ראשון שלילי", level="corona_result", axis=1).values.squeeze()
all_first = first.sum(axis=1).values.squeeze()
other_first = all_first - first_negative - first_positive

NotFirstTest = lab_tests.loc[lab_tests['is_first_Test'].str.contains('Yes') != True, ['result_date', 'corona_result']].reset_index()
not_first_grouped = NotFirstTest.groupby(['result_date', 'corona_result'],  as_index=False).count()
not_first = not_first_grouped.set_index(['result_date', 'corona_result']).unstack().fillna(0).astype(int).add_prefix('לא ראשון ')
del NotFirstTest, not_first_grouped
not_first_positive = not_first.xs("לא ראשון חיובי", level="corona_result", axis=1).values.squeeze()
not_first_negative = not_first.xs("לא ראשון שלילי", level="corona_result", axis=1).values.squeeze()
all_not_first = not_first.sum(axis=1).values.squeeze()
other_not_first = all_not_first - not_first_positive - not_first_negative

full_lab_data = pd.concat([first.squeeze(), not_first.squeeze()], axis=1, sort=False)
# Saving full data
full_lab_data.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), time.strftime("%d%m%Y")
                                  + 'complete_laboratory_data.csv'), encoding="windows-1255")

dateList = pd.DataFrame(lab_tests[base].unique(), columns=['Date'])

fields = ['PositiveFirst', 'NegativeFirst', 'OtherFirst', 'PositiveNotFirst', 'NegativeNotFirst', 'OtherNotFirst']

lab_data = pd.concat([dateList, pd.DataFrame(first_positive, columns=[fields[0]]),
                      pd.DataFrame(first_negative, columns=[fields[1]]),
                      pd.DataFrame(other_first, columns=[fields[2]]),
                      pd.DataFrame(not_first_positive, columns=[fields[3]]),
                      pd.DataFrame(not_first_negative, columns=[fields[4]]),
                      pd.DataFrame(other_not_first, columns=[fields[5]])],
                     axis=1, sort=False)

title = 'Israel ' + dateList.Date.max().strftime('%d/%m/%y') \
        + ' - count of first test per person. Total tests performed ' + num2strWithComma(int(N))
ylabel = 'Number of individuals'
save_name = 'israelTestPerformed_'
base = 'Date'
legend = ['Positive First test, total ' + num2strWithComma(int(lab_data.PositiveFirst.sum())),
          'Negative First test, total ' + num2strWithComma(int(lab_data.NegativeFirst.sum())),
          'Other First test, total ' + num2strWithComma(int(lab_data.OtherFirst.sum())),
          'Positive not a First test, total ' + num2strWithComma(int(lab_data.PositiveNotFirst.sum())),
          'Negative not a First test, total ' + num2strWithComma(int(lab_data.NegativeNotFirst.sum())),
          'Other not a First test, total ' + num2strWithComma(int(lab_data.OtherNotFirst.sum())), ]

# plot Test Performed Total
line_statistic_plot(lab_data, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)

# plot Test Performed Total Log
save_name = 'israelTestPerformed_Logy_'
line_statistic_plot(lab_data, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_logYes,
                    line_statistic_plot_fix_date)
del lab_tests


###################################################################################################################
# Individuals
individuals = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('tested_individuals_ver').values.argmax()])
individuals_last = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('tested_individuals_subset').values.argmax()])

id = files_db.current_file_name.str.find('tested_individual').values.argmax()
print([files_db.last_update[id], files_db.current_file_name[id], files_db.name[id]])
base = 'test_date'

individuals = individuals[individuals['test_date'].isna() != True]
N = len(individuals.corona_result)
individuals[base] = pd.to_datetime(individuals[base])
individuals = individuals.sort_values([base])
individuals_last[base] = pd.to_datetime(individuals_last[base])
individuals_last = individuals_last.sort_values([base])
individuals = individuals[(individuals['test_date'] >= individuals_last['test_date'].unique().min()).values != True]
individuals = pd.concat([individuals, individuals_last])
individuals['symptoms'] = individuals.loc[:, ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']].sum(axis=1)
possible_results = individuals.corona_result.unique()
dateList = pd.DataFrame(individuals[base].unique(), columns=['Date'])

# TestIndication
PosTest = individuals.loc[individuals['corona_result'].str.contains('חיובי') != False, ['test_date', 'test_indication']].reset_index()
posindicate = PosTest.groupby(['test_date', 'test_indication'],  as_index=False).count()
posindicate = posindicate.set_index(['test_date', 'test_indication']).unstack().fillna(0).astype(int)

# plot israelPositiveTestIndication
fields = ['Abroad', 'Contact with confirmed', 'Other']
title = 'Israel (data from ' + dateList.Date.max().strftime('%d/%m/%y') \
        + ')- Positive test indication (Total tests performed ' + str(int(N)) + ')'
ylabel = 'Number of positive tests'
save_name = 'israelPositiveTestIndication_'
Abroad = posindicate.xs('Abroad', level="test_indication", axis=1).values.squeeze()
Contact = posindicate.xs('Contact with confirmed', level="test_indication", axis=1).values.squeeze()
Other = posindicate.xs('Other', level="test_indication", axis=1).values.squeeze()
legend = ['Arrival from abroad, total ' + num2strWithComma(int(Abroad.sum())),
          'Contact with confirmed, total ' + num2strWithComma(int(Contact.sum())),
          'Other, total ' + str(int(Other.sum()))]
pos_indicate = pd.concat([dateList, pd.DataFrame(Abroad, columns=[fields[0]]),
                      pd.DataFrame(Contact, columns=[fields[1]]),
                      pd.DataFrame(Other, columns=[fields[2]])],
                      axis=1, sort=False)
line_statistic_plot(pos_indicate, 'Date', fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                    line_statistic_plot_fix_date)
del posindicate

# Run according to test indication results
name_possible_resuts = ['Other', 'Negative', 'Positive']
for ctest in range(len(possible_results)):
    test = possible_results[ctest]
    result = name_possible_resuts[ctest]
    # Syndromes
    syndromes = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']

    # Every Syndrome statistic per day
    PosSyindromes = individuals.loc[individuals['corona_result'].str.contains(test) != False, ['test_date', 'cough',
                                    'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']].reset_index()
    pos_synd_every = PosSyindromes.groupby(['test_date'], as_index=False).sum()
    # plot Positive Syndrome
    legend = syndromes
    fields = syndromes
    title = 'Israel Symptoms for ' + result + ' Result (data from ' + dateList.Date.max().strftime('%d/%m/%y') + ')'
    save_name = 'israel' + result + 'TestSymptoms_'
    ylabel = 'Symptoms'
    # usual plot
    line_statistic_plot(pos_synd_every, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                        line_statistic_plot_fix_date)
    # log plot
    save_name = 'israel' + result + 'TestSymptoms_Logy_'
    line_statistic_plot(pos_synd_every, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_logYes,
                        line_statistic_plot_fix_date)

    # Number of positive syndrome statistic per day
    PosSyindrome = individuals.loc[individuals['corona_result'].str.contains(test) != False, ['test_date', 'symptoms']].reset_index()
    pos_synd = PosSyindrome.groupby(['test_date', 'symptoms'], as_index=False).count()
    pos_synd = pos_synd.set_index(['test_date', 'symptoms']).unstack().fillna(0).astype(int)
    Noone = pos_synd.xs(0, level="symptoms", axis=1).values.squeeze()
    One = pos_synd.xs(1, level="symptoms", axis=1).values.squeeze()
    Two = pos_synd.xs(2, level="symptoms", axis=1).values.squeeze()
    Three = pos_synd.xs(3, level="symptoms", axis=1).values.squeeze()
    Four = pos_synd.xs(4, level="symptoms", axis=1).values.squeeze()
    Five = pos_synd.xs(5, level="symptoms", axis=1).values.squeeze()
    legend = ['No symptoms (asymptomatic), total ' + num2strWithComma(int(Noone.sum())),
              'One symptom, total ' + num2strWithComma(int(One.sum())),
              'Two symptoms, total ' + num2strWithComma(int(Two.sum())),
              'Three symptoms, total ' + num2strWithComma(int(Three.sum())),
              'Four symptoms, total ' + num2strWithComma(int(Four.sum())),
              'Five symptoms, total ' + num2strWithComma(int(Five.sum()))]
    fields = ['No', 'One', 'Two', 'Three', 'Four', 'Five']
    pos_syndrome = pd.concat([dateList, pd.DataFrame(Noone, columns=[fields[0]]),
                              pd.DataFrame(One, columns=[fields[1]]),
                              pd.DataFrame(Two, columns=[fields[2]]),
                              pd.DataFrame(Three, columns=[fields[3]]),
                              pd.DataFrame(Four, columns=[fields[4]]),
                              pd.DataFrame(Five, columns=[fields[5]])],
                             axis=1, sort=False)
    # plot Quantitative Symptoms
    title = 'Israel Quantitative Symptoms for ' + result + ' Result (data from ' + dateList.Date.max().strftime('%d/%m/%y') + ')'
    save_name = 'israelQuantitative' + result + 'TestSymptoms_'
    ylabel = 'Number of Symptoms'
    # usual plot
    line_statistic_plot(pos_syndrome, 'Date', fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                        line_statistic_plot_fix_date)
    # log plot
    save_name = 'israelQuantitative' + result + 'TestSymptoms_Logy_'
    line_statistic_plot(pos_syndrome, 'Date', fields, title, ylabel, legend, text, save_name,
                        line_statistic_plot_logYes, line_statistic_plot_fix_date)

    # More than two symptoms
    TwoPlus = Two + Three + Four + Five
    legend = ['No symptoms (asymptomatic), total ' + num2strWithComma(int(Noone.sum())),
              'One symptom, total ' + num2strWithComma(int(One.sum())),
              'Two + symptoms, total ' + num2strWithComma(int(TwoPlus.sum()))]
    fields = ['No', 'One', 'TwoPlus']
    pos_syndrome_compact = pd.concat([dateList, pd.DataFrame(Noone, columns=[fields[0]]),
                                      pd.DataFrame(One, columns=[fields[1]]),
                                      pd.DataFrame(TwoPlus, columns=[fields[2]])],
                                     axis=1, sort=False)
    # plot Quantitative Symptoms
    title = 'Israel Quantitative Symptoms for ' + result + ' Result (data from ' + dateList.Date.max().strftime('%d/%m/%y') + ')'
    save_name = 'israelQuantitative' + result + 'TestSymptoms_'
    ylabel = 'Number of Symptoms'
    # usual plot
    line_statistic_plot(pos_syndrome_compact, 'Date', fields, title, ylabel, legend, text, save_name)

    # No symptoms Ratio (asymptomatic)
    NooneRatio = Noone / (Noone + One + TwoPlus)
    OneRatio = One / (Noone + One + TwoPlus)
    TwoPlusRatio = TwoPlus / (Noone + One + TwoPlus)
    noSimRatioMean = NooneRatio[51:111].mean()
    legend = ['No symptoms Ratio (asymptomatic), total ' + num2strWithComma(int(Noone.sum())),
              'One symptom Ratio, total ' + num2strWithComma(int(One.sum())),
              'Two + symptoms Ratio, total ' + num2strWithComma(int(TwoPlus.sum()))]

    fields = ['NoR', 'OneR', 'TwoPlusR']
    pos_syndrome_compact_ratio = pd.concat([dateList, pd.DataFrame(NooneRatio*100, columns=[fields[0]]),
                                            pd.DataFrame(OneRatio*100, columns=[fields[1]]),
                                            pd.DataFrame(TwoPlusRatio*100, columns=[fields[2]])],
                                           axis=1, sort=False)
    # plot Quantitative Symptoms
    title = 'Israel Quantitative Symptoms Ratio for ' + result + ' Result (data from ' \
            + dateList.Date.max().strftime('%d/%m/%y') + ')'
    save_name = 'israelQuantitativeRatio' + result + 'TestSymptoms_'
    ylabel = 'Symptoms per patient [%]'
    # usual plot
    line_statistic_plot(pos_syndrome_compact_ratio, 'Date', fields, title, ylabel, legend, text, save_name,
                        line_statistic_plot_log, line_statistic_plot_fix_date)

    print('Asymptomatic ' + str(int(Noone.sum()/(Noone.sum() + One.sum() + TwoPlus.sum()) * 100)) + '% ' + result)


# Comparison between WHO and Israel Data
###################################################################################################################
try:
    try:
        db = pd.read_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'israel_db.csv'))
    except:
        db = pd.read_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'Israel_db.csv'))
    db['Date'] = pd.to_datetime(db['Date'])
    if db.Date.max() <= lab_data.Date.max():
        lab_data = lab_data[(lab_data['Date'] >= db.Date.max()).values != True]
        pos_indicate = pos_indicate[(pos_indicate['Date'] >= db.Date.max()).values != True]
    else:
        db = db[(db['Date'] >= lab_data.Date.max()).values != True]
    individ = pd.DataFrame(pos_indicate.iloc[:, 1:].sum(axis=1).values, columns=['Individ'])
    compare_db = pd.concat([db[['Date', 'NewConfirmed']], lab_data['PositiveFirst'], individ['Individ']], axis=1, sort=False)

    save_name = 'newCasesWHODataVsIsraelData'
    title = 'Israel New Confirmed WHO data vs. Israel Ministry of Health data'
    ylabel = 'Number of individuals'
    legend = ['New cases WHO data, total ' + num2strWithComma(int(db.NewConfirmed.sum())) + ' at '
              + db.Date.max().strftime('%d/%m/%y'), 'Positive first test from lab_tests.csv data, total '
              + num2strWithComma(int(lab_data.PositiveFirst.sum())) + ' at '
              + dateList.Date.max().strftime('%d/%m/%y'), 'Positive test from tested_individuals.csv data, total '
              + num2strWithComma(int(individ['Individ'].sum())) + ' at ' + dateList.Date.max().strftime('%d/%m/%y')]

    fields = ['NewConfirmed', 'PositiveFirst', 'Individ']
    title = 'Israel Symptoms for ' + result + ' Result (data from ' + dateList.Date.max().strftime('%d/%m/%y') + ')'
    base = 'Date'
    line_statistic_plot(compare_db, base, fields, title, ylabel, legend, text, save_name, line_statistic_plot_log,
                        line_statistic_plot_fix_date)
except:
    print('The file israel_db.csv is not loaded')


###################################################################################################################
# Load Geographical data of Israel Cities
url = 'https://opendata.arcgis.com/datasets/a589d87604c6477ca4afb78f205b98fb_0.geojson'
r = requests.get(url)
data = json.loads(r.content)
df = pd.json_normalize(data, ['features'])
z = df.pop('type')
zg = df.pop('geometry.type')
id_names = ['OBJECTID_1', 'OBJECTID', 'SETL_CODE', 'MGLSDE_LOC', 'MGLSDE_L_1', 'MGLSDE_L_2', 'MGLSDE_L_3',
            'MGLSDE_L_4', 'coordinates'], df.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"),
                                                                 'IsraelCitiesCoordinates.csv'), encoding="ISO-8859-8")

geographic = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('geographic').values.argmax()])
# new_recoveries_on_date	accumulated_hospitalized	new_hospitalized_on_date
# accumulated_deaths	new_deaths_on_date	accumulated_diagnostic_tests
# there are two files with city in name:
# 12: corona_city_table_ver_0055.csv
# 15: vaccinated_city_table_ver_0013.csv
# which one TODO?
# cities = pd.read_csv(files_db.current_file_path[files_db.current_file_name.str.find('city')])
# City_Name	City_Code	Date	Cumulative_verified_cases	Cumulated_recovered	Cumulated_deaths
# Cumulated_number_of_tests	Cumulated_number_of_diagnostic_tests	Cumulated_vaccinated

geo = geographic.groupby(['town_code'],  as_index=False)
# city = cities.groupby(['City_Code'],  as_index=False)

# TODO some calculations
# TODO create map with data

# City_Color = NewConfirmed * PositiveTestPercent * sickness_growth_rate
# red    = [7.5 10.0]
# orange = [6.0 7.5]
# yellow = [4.5 6.0]
# green  = [0.0 4.5]

# R coefficient according Cori et al: https://academic.oup.com/aje/article/178/9/1505/89262
# r(t) = NewConfirmed(t) / (NewConfirmed(t-3)*G(t-3) + ... + NewConfirmed(t+3)*G(t+3))
# G - gamma distribution with mean=4.5 and std=3.5 - generation interval distribution
###################################################################################################################

plt.close('all')

fout.close()
# sys.stdout.close()
sys.stdout = stdoutOrigin