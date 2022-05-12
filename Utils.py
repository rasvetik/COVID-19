"""
Created on Thursday Mar 26 2020
Sveta Raboy
based on
https://www.kaggle.com/bardor/covid-19-growing-rate
https://github.com/CSSEGISandData/COVID-19
https://github.com/imdevskp
https://www.kaggle.com/yamqwe/covid-19-status-israel

"""

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import folium
import plotly
import os
import time
import matplotlib.dates as mdates

# plt.style.use('dark_background')


# Smooth Vector with Mean By N
def smooth_vector(v, n=4):
    v1 = v.copy()
    for i in range(n, len(v1) - (n - 1)):
        v1[i] = v[i - (n - 1):i + n].sum() / (n + (n - 1))
    return v1


# Write Log file
class MyWriter:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)

    def flush(self):
        for w in self.writers:
            w.flush()


# bar plot
def bar_country_plot(full_data, groupby='Date', inputs=['Confirmed', 'Active', 'Recovered', 'Deaths'],
                     fname='_cases_bars', log=False):
    # Confirmed vs Recovered and Death
    if isinstance(full_data.Date.max(), str):
        day = datetime.datetime.strptime(full_data.Date.max(), '%m/%d/%y').strftime('%d%m%y')
    else:
        day = full_data.Date.max().strftime('%d%m%y')

    title_string = full_data.State + ' Cases' + ' for' + day
    with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day + '_' + full_data.State + '_' + fname + '.html'), 'a') as ff:
        fig = px.bar(full_data, x=groupby, y=inputs, color=inputs, template='ggplot2', log_y=True,
                     title=title_string, hover_name=inputs)
        fig.layout.template = 'plotly_dark'
        # fig.show()
        ff.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%'))

    f = plt.figure(figsize=(9, 7))
    colors = ['blue', 'green', 'cyan', 'magenta', 'cyan', 'red', 'black']
    alphas = [1, 0.75, 0.75, 1]
    title_string = str()

    for cnt in range(len(inputs)):
        k = inputs[cnt]
        plt.bar(full_data[groupby], full_data[k], label=k, alpha=alphas[cnt], log=log, color=colors[cnt])
        title_string = title_string + k + ' vs '

    plt.xlabel('Date')
    plt.ylabel("Count")
    plt.legend(frameon=True, fontsize=12)
    plt.title(title_string[:-4], fontsize=30)
    f.autofmt_xdate()
    plt.show()

    plt.savefig(os.path.join(os.getcwd(), day + '_' + str(full_data['Country'].unique().values) + '.png'))
    return f
##############################################################################################


# Normalise
def normalise_func(input_data, inputs=['Confirmed', 'Deaths', 'Recovered', 'Active'], name='NormPop',
                   normaliseTo='Population', factor=1e6, toRound=False):

    for cnt in range(len(inputs)):
        k = inputs[cnt]
        new_name = name+k
        input_data.loc[:, new_name] = 0
        # Normalise to Population with factor of 1M
        input_data.loc[:, new_name] = (input_data[k].values * factor / (input_data[normaliseTo].values + 1e-6)).clip(0)
        if toRound:
            input_data.loc[input_data.loc[:, new_name] > 1, new_name] = input_data.loc[input_data.loc[:, new_name] > 1, new_name].astype(int)

    return input_data
############################################################################################################


# Events
def add_events(input_data, events):
    input_data.loc[:, 'Event'] = ''
    for cnt in range(events.shape[0]):
        input_data.loc[input_data['Date'] == events.Date[cnt], 'Event'] = events.Event[cnt]
    return input_data
######################################################################################################


# Growth
def growth_func(input_data, inputs, numDays=1, name='Growth', normalise=True, prediction_Range=1):

    for cnt in range(len(inputs)):
        k = inputs[cnt]
        input_data.loc[:, name+k] = 0
        if normalise:
            input_data.loc[:, name+k] = ((input_data[k] / input_data[k].shift(numDays)) ** prediction_Range - 1) * 100.0  # .clip(0)
            input_data.loc[input_data[k].shift(-numDays) == 0, name+k] = 0
        else:
            input_data[name+k] = (input_data[k] - input_data[k].shift(numDays))  # .clip(0)

    return input_data
############################################################################################################


# add the population and age columns for the given data
def add_pop_age_data(input_data, world_population):
    world_pop = None
    try:
        input_data.loc[:, 'Population'] = np.nan
        input_data.loc[:, 'Age'] = np.nan
    except:
        input_data['Population'] = np.nan
        input_data['Age'] = np.nan

    for val in input_data.Country.unique():
        curr = world_population[world_population['Country'] == val]
        cntries = input_data.Country == val
        try:
            input_data.loc[cntries, 'Population'] = curr['Population'].values
            input_data.loc[cntries, 'Age'] = curr['Age'].values
            if world_pop is not None:
                world_pop = pd.concat([world_pop, curr], axis=0, sort=False)
            else:
                world_pop = curr
        except ValueError:
            pass
    return input_data, world_pop
#########################################################################################


# extract data according to group(Date and State) and if flag add_value is True add the country value to string of State
def group_extract_data(full_data, world_population, groupby=['Date', 'State', 'Country'], inputs=['Confirmed'],
                       threshould=5000, add_value=True):
    sorted_data = full_data.sort_values(groupby)
    group = sorted_data[groupby[1]].unique()
    latest = sorted_data[sorted_data.Date == sorted_data.Date.max()]
    remain_data = latest[latest[inputs] > threshould][groupby[1]].unique()
    relevant = sorted_data.copy()
    for val in group:
        if (remain_data != val).all():
            relevant = relevant[relevant[groupby[1]].str.endswith(val) != True]
        elif not relevant[groupby[2]].str.endswith(val).any() and add_value:
            relevant.loc[relevant[groupby[1]].str.endswith(val), groupby[1]] = \
                relevant.loc[relevant[groupby[1]].str.endswith(val), groupby[2]].values[0] + \
                '_' + val

    relevant, world_pop = add_pop_age_data(relevant, world_population)

    return relevant, world_pop
################################################################################################


# Create Sum Table
def create_table(indata, day, inputs=['Confirmed', 'Deaths', 'Recovered', 'Active'],
                 h_columns=['Current Day', 'Total', 'Max Value'], title_string='', height='100%',
                 fname='_World_Daily_Situation_Summarise_Table'):

    head = indata[inputs].keys().values.tolist()
    head.insert(0, h_columns[0])
    body = [h_columns[1:]]
    for cnt in range(len(inputs)):
        body.append(indata[inputs[cnt]].values)

    with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + fname + '.html'), 'a') as f:
        fig = go.Figure(data=[go.Table(header=dict(values=head, height=35, align=['left', 'center']),
                                       cells=dict(values=body, height=28, align='left'))])
        fig.layout.template = 'plotly_dark'
        fig.layout.title = day.strftime('%d/%m/%y ') + title_string
        # fig.show()
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=height))
########################################################################################################


# Create countries bar
def countries_bar(indata, day, groupby=['Country'], inputs=None, count=30, fname='_World_Daily_Situation'):
    if inputs is None:
        inputs = indata.keys()[1:].values
    with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + fname + '.html'), 'a') as f:
        for cnt in range(len(inputs)-1, -1, -1):
            k = inputs[cnt]
            cur_data = indata.sort_values(k, ascending=0).reset_index()
            cur_data = cur_data[:count]
            if k == 'Population' or k == 'Age':
                add_str = ''
            else:
                add_str = ' Cases'
            if cnt in range(4):
                f_str = 'Total '
            else:
                f_str = ''
            title_string = f_str + k + add_str + ' for ' + day.strftime('%d/%m/%y') + ': ' + str(count) \
                                 + ' countries from ' + str(indata.shape[0])
            fig = px.bar(cur_data, x=groupby[0], y=k, color=groupby[0], text=k, template='ggplot2', log_y=True,
                         title=title_string)  # , hover_name=groupby[0])
            fig.layout.template = 'plotly_dark'
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            # fig.show()

            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


# Create World Map
def create_map(data, world_pop, location=[31, 35]):
    # Israel location start
    # Affected place in world map including Confirm , Active,  Deaths and Recovery
    worldmap = folium.Map(location=location, zoom_start=4, tiles='Stamen Terrain')

    for lat, long, country, state, conf, death, recover, active in zip(data['Lat'], data['Long'], data['Country'],
                                                                       data['State'], data['Confirmed'], data['Deaths'],
                                                                       data['Recovered'], data['Active']):

        cur_pop = world_pop[world_pop['Country'] == country].reset_index()
        if isinstance(state, str) and state != country or not cur_pop.sum().any():
            popup_str = str(country) + '<br>' + 'State: ' + str(state) + '<br>' +\
                        'PositiveCases:' + str(conf) + '<br>' +\
                        'Active:' + str(int(active)) + '<br>' +\
                        'Recovered:' + str(int(recover)) + '<br>' +\
                        'Deaths:' + str(death) + '<br>'
        elif np.isnan(cur_pop['Age'][0]):
            popup_str = str(country) + ' Population:' + str(cur_pop['Population'][0]) + '<br>'\
                        'Positive:' + str(conf) + '<br>' + \
                        'Active:' + str(int(active)) + '<br>' + \
                        'Recovered:' + str(int(recover)) + '<br>' + \
                        'Deaths:' + str(death) + '<br>'
        else:
            popup_str = str(country) + ' Population:' + str(cur_pop['Population'][0]) + \
                        ' Median Age:' + str(int(cur_pop['Age'][0])) + '<br>' + \
                        'Positive:' + str(conf) + '<br>' + \
                        'Active:' + str(int(active)) + '<br>' + \
                        'Recovered:' + str(int(recover)) + '<br>' + \
                        'Deaths:' + str(death) + '<br>'
        folium.CircleMarker([lat, long], radius=5, color='red', popup=popup_str, fill_color='red',
                            fill_opacity=0.7).add_to(worldmap)
    # in IPython Notebook, Jupyter
    worldmap
    day = data.Date.max().strftime('%d%m%y')
    worldmap.save(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day + '_WorldMap.html'))
###################################################################################################


# bar plot according to cases
def case_groupby_bar(full_data, world_population, groupby=['Date', 'State', 'Country'],
                     inputs=['Confirmed', 'Recovered', 'Deaths', 'Active'], threshould=[10000, 1000, 100, 10000],
                     normalise=True, fname='_Cases_WorldData_Bars', factor=1e6):

    daily = full_data.sort_values(groupby)
    states = daily[groupby[1]].unique()
    day = full_data.Date.max().strftime('%d/%m/%y')
    array_relevant = []

    for cnt in range(len(inputs)):
        k = inputs[cnt]
        with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), full_data.Date.max().strftime('%d%m%y') + '_' + k + fname + '.html'), 'a') as f:
            relevant, world_pop = group_extract_data(daily, world_population, groupby, k, threshould[cnt])
            array_relevant.append(relevant)
            srelevant = relevant.sort_values([groupby[0], groupby[1], k], ascending=[1, 1, 0])
            srelevant.Date = [datetime.datetime.strftime(d, '%d/%m/%Y') for d in srelevant.Date]
            num_contries = len(relevant[groupby[1]].unique())
            title_string = k + ' Cases' + ' over ' + str(threshould[cnt]) + ' for ' + day + ': ' \
                             + str(num_contries) + ' items from ' + str(len(states))
            fig = px.bar(srelevant, y=groupby[1], x=k, color=groupby[1], template='ggplot2', orientation='h',
                         log_x=True, title=title_string, hover_name=groupby[1], animation_frame=groupby[0],
                         animation_group=groupby[1])
            fig.layout.template = 'plotly_dark'
            # soup = BeautifulSoup(ff)
            height = str(np.max([100, num_contries/25 * 100])) + '%'
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height=height))
            # in IPython Notebook, Jupyter, etc
            # fig.show()
            # Another way to save
            # fig.write_html(os.path.join(os.getcwd(), full_data.Date.max().strftime('%d%m%y') + '_WorldData.html'))
            del fig

            if normalise:
                # Normalise to Population with factor of 1M
                norm_srelevant = srelevant.copy()
                norm_srelevant.loc[:, k] = (norm_srelevant[k].values * factor /
                                            norm_srelevant['Population'].values).clip(0)
                norm_srelevant.loc[norm_srelevant.loc[:, k] > 1, k] = norm_srelevant.loc[norm_srelevant.loc[:, k] > 1, k].astype(int)
                num_contries = len(relevant[groupby[1]].unique())
                title_string = k + ' Cases' + ' over ' + str(threshould[cnt]) + ' Normalized to ' + str(int(factor/1e6)) \
                                 + 'M population' + ' for ' + day + ': ' + str(num_contries) + ' items from ' \
                                 + str(len(states))
                fig = px.bar(norm_srelevant, y=groupby[1], x=k, color=groupby[1], template='ggplot2', log_x=True,
                             orientation='h', title=title_string, hover_name=groupby[1], animation_frame=groupby[0],
                             animation_group=groupby[1])
                fig.layout.template = 'plotly_dark'
                height = str(np.max([100, num_contries/25 * 100])) + '%'
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height=height))
                del fig

                # Normalised to inputs[0]: Confirmed
                if cnt > 0:
                    # probability of dying/ recovered if infected by the virus (%)
                    norm_srelevant = srelevant.copy()
                    norm_srelevant.loc[:, k] = (norm_srelevant[k].values /
                                                (norm_srelevant[inputs[0]].values + 1e-6)).clip(0)
                    norm_srelevant.loc[norm_srelevant[k] > 1, k] = 1
                    num_contries = len(relevant[groupby[1]].unique())
                    title_string = k + ' Cases' + ' over ' + str(threshould[cnt]) + ' Normalized to ' + inputs[0] \
                                     + ' for ' + day + ': ' + str(num_contries) + ' items from ' + str(len(states))\
                                     + '<br>"Probability" of ' + k + ' If Infected by the Virus'
                    fig = px.bar(norm_srelevant, y=groupby[1], x=k, color=groupby[1], template='ggplot2',
                                 orientation='h', title=title_string, hover_name=groupby[1], animation_frame=groupby[0],
                                 animation_group=groupby[1])
                    fig.layout.template = 'plotly_dark'
                    height = str(np.max([100, num_contries/25 * 100])) + '%'
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height=height))
                    del fig
#################################################################################################


# scatter plot
def scatter_country_plot(full_data, inputs=['Confirmed', 'Recovered', 'Deaths', 'Active'], base='Date', prefix='',
                         fname=' Total Cases ', add_growth_rates=False, num_days_for_rate=14, annotations=None,
                         add_events_text=False, factor=1.0, mat_plt=False, day=''):

    if not day:
        if isinstance(full_data.Date.max(), str):
            day = datetime.datetime.strptime(full_data.Date.max(), '%m/%d/%y').strftime('%d%m%y')
        else:
            day = full_data.Date.max().strftime('%d/%m/%y')

    try:
        not_country = 0
        country = full_data['Country'].unique()
        state = full_data['State'].unique()
    except:
        not_country = 1

    if not_country or country.shape[0] > 1:
        title_string = day + fname + 'Various Cases'
        save_string = full_data.Date.max().strftime('%d%m%y') + fname + '.png'
    elif state != country:
        title_string = country[0] + ' -- ' + state[0] + ' - ' + day + ' ' + fname
        save_string = full_data.Date.max().strftime('%d%m%y') + '_' + country[0] + '_' + state[0] + '_' +\
                      fname.replace(' ', '_') +'.png'
    else:
        title_string = state[0] + ' - ' + day + ' - ' + fname
        save_string = full_data.Date.max().strftime('%d%m%y') + '_' + state[0] + '_' + fname.replace(' ', '_') +'.png'

    # colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    colors = plotly.colors.qualitative.Light24
    if '#FED4C4' in colors:
        colors.remove('#FED4C4')
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Linear Plot", "Log Plot"))
    fig_cnt = -1
    customdata = None

    for cnt in range(len(inputs)):
        case_k = inputs[cnt]
        k = prefix + case_k
        y = (full_data[k] * factor).fillna(0)
        # y[np.isinf(y)] = 0
        if base != 'Date':
            customdata = full_data.Date
        if add_events_text:
            trace = go.Scatter(x=full_data[base], y=y, mode="markers+lines+text", name=case_k, customdata=customdata,
                               text=full_data.Event, marker=dict(size=8, color=colors[cnt]))
        else:
            trace = go.Scatter(x=full_data[base], y=y, mode="markers+lines", name=case_k, customdata=customdata,
                               marker=dict(size=8, color=colors[cnt]))
        fig.add_trace(trace, row=1, col=1)
        fig_cnt +=1
        fig.add_trace(trace, row=1, col=2)
        fig_cnt += 1
        if fig_cnt % 2 == 1:
            fig.data[fig_cnt-1].update(showlegend=False)

        fig.update_traces(mode="markers+lines", hovertemplate=None)
        if base != 'Date':
            fig.update_traces(hovertemplate='%{y}<br>%{customdata| %_d %b %Y}')

        if add_growth_rates:
            len_rate = full_data[k].shape[0]
            grows_rate = full_data['Growth' + base].fillna(0).values / 100.0
            grows_rate[np.isinf(grows_rate)] = 0
            vec = np.arange(0, round(len_rate*1/3))
            one_third = grows_rate[vec].mean()
            if one_third > 0:
                grow_one_third = one_third * full_data[base] + full_data[k][vec[0]] * factor
                add_trace1 = go.Scatter(x=full_data[base], y=grow_one_third, mode="lines",
                                        name='Linear estimation: ' + str(full_data[k][vec[0]]) + ' + '
                                             + str(round(one_third, 3)) + '*' + base + '<br>' + str(round(one_third, 3))
                                        + ' - estim on first onethird of ' + base,
                                        line=dict(dash="dash", width=3))
                fig.add_trace(add_trace1, row=1, col=1)
                fig.add_trace(add_trace1, row=1, col=2)

            # estimation for two last weeks
            vec = np.arange(np.max([1, len_rate-num_days_for_rate]), len_rate)
            last_week = (full_data[k][vec[-1]] - full_data[k][vec[0]]) \
                        / np.max([1e-6, (full_data[base][vec[-1]] - full_data[base][vec[0]])])
            if not np.isinf(last_week) and last_week > 0:
                bias = int(full_data[k][vec[-1]] - full_data[base][vec[-1]] * last_week)
                grow_one_third = last_week * full_data[base] + bias * factor
                add_trace2 = go.Scatter(x=full_data[base][round(len_rate*1/3):], y=grow_one_third[round(len_rate*1/3):],
                                        mode="lines", name='Linear estimation: ' + str(bias) + ' + '
                                                           + str(round(last_week, 3)) + '*' + base + '<br>'
                                                           + str(round(last_week, 3)) + ' - estim on '
                                                           + str(num_days_for_rate) + ' last days from ' + base,
                                        line=dict(dash="dash", width=3))
                fig.add_trace(add_trace2, row=1, col=1)
                fig.add_trace(add_trace2, row=1, col=2)
            fig.update_yaxes(range=[full_data[k][0], full_data[k][len_rate-1]], row=1, col=1)

    if annotations is not None:
        fig.update_annotations(annotations)

    fig.update_layout(template='plotly_dark', hovermode="x", title=title_string,
                      yaxis=dict(title=fname), xaxis=dict(title=base), yaxis2=dict(title=fname, type='log'),
                      xaxis2=dict(title=base))
    # fig.show()
    if mat_plt:
        fig_mat, ax = plt.subplots(figsize=(8, 6))
        colors = ['blue', 'green', 'yellow', 'magenta', 'cyan', 'red', 'black']
        max_values = []
        for cnt in range(len(inputs)):
            case_k = inputs[cnt]
            k = prefix + case_k
            full_data[k] = full_data[k].fillna(0)
            ax = sns.scatterplot(x=base, y=k, data=full_data, color=colors[cnt])
            plt.plot(full_data[base], full_data[k], zorder=1, color=colors[cnt], label=k)
            if not np.isinf(max(full_data[k])):
                max_values.append(max(full_data[k]))
        ax.set_xlim([full_data['Date'].iloc[0],  full_data['Date'].iloc[-1] + datetime.timedelta(days=1)])

        if max(full_data[prefix + inputs[0]]) > 1:
            max_value = max(max_values) + np.diff(full_data[k]).max()
            min_value = -1
        else:
            max_value = max(max_values) + np.diff(full_data[k]).max()
            min_value = 0
        ax.set_ylim([min_value, max_value])
        plt.legend(frameon=True, fontsize=12)
        plt.grid()
        plt.ylabel(fname)
        plt.title(title_string, fontsize=16)
        fig_mat.autofmt_xdate()
        plt.savefig(os.path.join(os.getcwd(), save_string))

    return fig
###################################################################################################################


# country analysis script
def country_analysis(clean_db, world_pop, country='China', state='Hubei', plt=False, fromFirstConfirm=False,
                     events=None, num_days_for_rate=14):
    if isinstance(clean_db.Date.max(), str):
        day = datetime.datetime.strptime(clean_db.Date.max(), '%m%d%y').strftime('%d%m%y')
    else:
        day = clean_db.Date.max().strftime('%d%m%y')

    data = clean_db[clean_db['Country'] == country]
    data = data.sort_values(by='Date', ascending=1)
    today = data.Date.iloc[-1].strftime('%d.%m.%y')
    if state:
        data = data[data['State'] == state]
    elif (data.State.unique() == country).any():
        data = data[data['State'] == country]
    else:
        data = data.groupby(['Date', 'Country']).sum()

    if fromFirstConfirm:
        data = (data.loc[data.loc[:, 'Confirmed'] > 0, :]).reset_index()
    else:
        data = data.reset_index()

    data['Active'] = (data['Confirmed'] - data['Recovered'] - data['Deaths']).astype(int)  # .clip(0)
    inputs = ['Confirmed', 'Recovered', 'Deaths', 'Active']
    data = growth_func(data, inputs, numDays=1, name='New', normalise=False)
    data = growth_func(data, inputs, numDays=1, name='Growth', normalise=True)

    cur_pop_data = world_pop[world_pop['Country'] == country].reset_index()
    data.loc[:, 'Population'] = cur_pop_data['Population'].values[0]
    data.loc[:, 'Age'] = cur_pop_data['Age'].values[0]

    data = normalise_func(data, name='NormPop', normaliseTo='Population', factor=1e6, toRound=True)
    data = normalise_func(data, inputs=['Deaths', 'Recovered', 'Active'], name='NormConfirm', normaliseTo='Confirmed',
                          factor=1, toRound=True)
    add_event = False
    if events is not None:
        data = add_events(data, events)
        add_event = True
    # Growth Rate
    # last_days = data['Confirmed'].shift()[-3:]
    # gr = data['Confirmed'][-3:] / last_days
    # gr[last_days == 0] = 0
    growth_rate = (data['Confirmed'][-3:] / data['Confirmed'].shift()[-3:]).fillna(0).mean()
    growth_death = (data['Deaths'][-3:] / data['Deaths'].shift()[-3:]).fillna(0).mean()
    growth_recovered = (data['Recovered'][-3:] / data['Recovered'].shift()[-3:]).fillna(0).mean()
    prediction_cnfm = 0
    prediction_dth = 0
    prediction_rcv = 0
    expected_cnfrm = 0
    expected_dth = 0
    expected_rcv = 0
    if growth_rate != 0 and growth_rate != 1 and not np.isinf(growth_rate):
        prediction_cnfm = (np.log(2)/np.log(growth_rate)).clip(0).astype(int)
        expected_cnfrm = (data['Confirmed'].iloc[-1] * growth_rate).astype(int)
    if growth_death != 0 and growth_death != 1 and not np.isinf(growth_death):
        prediction_dth = (np.log(2)/np.log(growth_death)).clip(0).astype(int)
        expected_dth = (data['Deaths'].iloc[-1] * growth_death).astype(int)
    if growth_recovered != 0 and growth_recovered != 1 and not np.isinf(growth_recovered):
        prediction_rcv = (np.log(2)/np.log(growth_recovered)).clip(0).astype(int)
        expected_rcv = (data['Recovered'].iloc[-1] * growth_recovered).astype(int)

    print('\n', country)
    print('Mean Growth Rate for 3 last days : Confirmed %.2f%%, Deaths %.2f%%, Recovered %.2f%%'
          % (round((growth_rate-1)*100.0, 2), round((growth_death-1)*100.0, 2), round((growth_recovered-1)*100.0, 2)))
    print('Today\'s %s   [confirmed, death, recovered] :  %d, %d, %d ' % (today, data['Confirmed'].iloc[-1],
          data['Deaths'].iloc[-1], data['Recovered'].iloc[-1]))
    print('Expected Tomorrow      [confirmed, death, recovered] :  %d, %d, %d ' %
          (expected_cnfrm, expected_dth, expected_rcv))
    #  logarithm of x to the given base, calculated as log(x)/log(base)
    days = [prediction_cnfm, prediction_dth, prediction_rcv]
    print('Twice the number of cases given the current growth rate in %s days' % days)

    annot = dict(xref='paper', yref='paper', x=0.2, y=0.95, align='left', font=dict(size=12),
                 text='Mean Growth Rate for 3 last days:  Confirmed ' + str(round((growth_rate-1)*100.0, 2))
                      + '%,  Deaths ' + str(round((growth_death-1)*100.0, 2)) + '%,  Recovered '
                      + str(round((growth_recovered-1)*100.0, 2))
                      + '%<br>Today\'s ' + str(today) + '   [confirmed, death, recovered] :   '
                      + str(data['Confirmed'].iloc[-1]) + '   ' + str(data['Deaths'].iloc[-1]) + '   '
                      + str(data['Recovered'].iloc[-1].astype(int))
                      + '<br>Expected Tomorrow     [confirmed, death, recovered] :   '
                      + str(expected_cnfrm) + '   ' + str(expected_dth) + '   ' + str(expected_rcv)
                      + '<br>Twice the number of cases given the current growth rate in   '
                      + str(prediction_cnfm) + '   ' + str(prediction_dth) + '   ' + str(prediction_rcv) + '  days')
    if plt:
        if country[-1] == '*':
            country = country[:-1]
        with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day + '_' + country + '_Various_Cases.html'), 'a') as f:
            fsc1 = scatter_country_plot(data, add_events_text=add_event)
            fsc2 = scatter_country_plot(data, prefix='New', fname='Daily New Cases', add_events_text=add_event)
            fsc3 = scatter_country_plot(data, prefix='NormPop', fname='Total Cases Normalised for 1M Population',
                                        add_events_text=add_event)
            fsc4 = scatter_country_plot(data, inputs=['Deaths', 'Recovered', 'Active'], prefix='NormConfirm',
                                        factor=100.0, add_events_text=add_event,
                                        fname='Normalised for Total Confirmed Cases - '
                                              'Probability to Case If infected by the virus (%)')
            fsc5 = scatter_country_plot(data, prefix='Growth', add_events_text=add_event,
                                        fname='Growing rate in % a day', annotations=annot)
            fsc6 = scatter_country_plot(data, inputs=['Deaths'], add_events_text=add_event, base='Recovered',
                                        add_growth_rates=True, num_days_for_rate=num_days_for_rate,
                                        fname='Cases Ratio: Deaths vs Recovered')

            f.write(fsc1.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fsc2.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fsc3.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fsc4.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fsc5.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fsc6.to_html(full_html=False, include_plotlyjs='cdn'))
    return data
###########################################################################################################


# plot with threshoulds on cases
def case_thresh_plot(full_data, threshDays=[10, 10], inputs=['Confirmed', 'Deaths'], prefix='', ref_cntry='Israel',
                     base='Date', factor=1.0, fname=' Corona virus situation since the ', annotations=[], log=False,
                     add_growth_rates=False, threshValues=[1, 1]):
    if isinstance(full_data.Date.max(), str):
        day = datetime.datetime.strptime(full_data.Date.max(), '%m/%d/%y').strftime('%d%m%y')
    else:
        day = full_data.Date.max().strftime('%d%m%y')
    countries = full_data.Country.unique()
    today = full_data.Date.iloc[-1].strftime('%d.%m.%y')

    title_string = full_data.Date.max().strftime('%d/%m/%y') + ' - ' + str(len(countries)) + ' ' + fname
    colors = plotly.colors.qualitative.Light24
    if '#FED4C4' in colors:
        colors.remove('#FED4C4')
    ref_db = full_data[full_data.Country == ref_cntry]
    ref_db = ref_db.sort_values([base])

    fig = make_subplots(rows=1, cols=2, subplot_titles=(prefix + ' ' + inputs[0] + ' Cases',
                                                        prefix + ' ' + inputs[1] + ' Cases'))

    showlegend = True

    for cnt in range(len(inputs)):
        case_k = inputs[cnt]
        k = prefix + case_k
        threshDay = threshDays[cnt]
        threshValue = threshValues[cnt]
        max_value = []
        customdata = None

        if cnt % 2:
            showlegend = False

        for cntry in range(len(countries)):
            curr = full_data[full_data.Country == countries[cntry]]
            thresh_data = curr.loc[curr.loc[:, k] * factor > threshValue, :]
            thresh_data = thresh_data[threshDay:]
            if thresh_data.values.any():
                thresh_data = thresh_data.sort_values([base, k])
                max_value.append(thresh_data[k].max())
                customdata = thresh_data[base]
                since_days = np.arange(0, thresh_data.shape[0])
                trace = go.Scatter(x=since_days, y=thresh_data[k], mode="markers+lines", name=countries[cntry],
                                   marker=dict(size=10, color=colors[cntry]), showlegend=showlegend, customdata=customdata)
                fig.add_trace(trace, row=1, col=cnt+1)

    fig.update_traces(hovertemplate=None)

    fig.update_traces(hovertemplate='%{y}<br>%{customdata| %_d %b %Y}')

    if add_growth_rates:
        for cnt in range(len(inputs)):
            case_k = inputs[cnt]
            k = prefix + case_k
            threshDay = threshDays[cnt]
            threshValue = threshValues[cnt]
            showlegend = True
            if cnt % 2:
                showlegend = False
            threshed_ref_db = ref_db.loc[ref_db.loc[:, k] * factor > threshValue, :]
            threshed_ref_db = threshed_ref_db[threshDay:]

            if threshed_ref_db.values.any():
                if 'Growth' + k not in threshed_ref_db.keys():
                    threshed_ref_db = growth_func(threshed_ref_db, [k])
                    grows_rate = threshed_ref_db['Growth' + k].fillna(0).values / 100.0 + 1
                    grows_rate[np.isinf(grows_rate)] = 0
                    growth_rate_mean = grows_rate[-3:].mean()
            else:
                threshed_ref_db = thresh_data.copy()
                growth_rate_mean = (threshed_ref_db[k][-3:] / threshed_ref_db[k].shift()[-3:]).fillna(0).mean()  # .clip(0)

            if growth_rate_mean != 0 and growth_rate_mean != 1 and not np.isinf(growth_rate_mean) and not np.isnan(growth_rate_mean):
                gr_days = (np.log(2) / np.log(growth_rate_mean)).astype(int)
                prev_value = threshed_ref_db[k].iloc[-2].astype(int)
                next_value = (threshed_ref_db[k].iloc[-1] * growth_rate_mean).astype(int)
            else:
                gr_days = 0
                prev_value = 0
                next_value = 0
                growth_rate_mean = 0

            if gr_days:
                annot = dict(xref='paper', yref='paper', x=0.2 + cnt*0.55, y=0.87, align='left', font=dict(size=13),
                             text='Mean Growth Rate for 3 last days in ' + threshed_ref_db.Country.values[0] + ' :  '
                                  + str(round((growth_rate_mean - 1) * 100.0, 2))
                                  + '%<br>Today\'s ' + str(today) + ' ' + inputs[cnt] + ':  ' + str(prev_value)
                                  + '<br>Expected Tomorrow: ' + str(next_value)
                                  + '<br>Twice the number of cases given the current growth rate in  ' + str(gr_days)
                                  + ' days')
                fig.add_annotation(annot)

            num_dates = threshed_ref_db[base].shape[0]
            if num_dates:
                since_days = np.arange(0, threshed_ref_db.shape[0])
                max_value.append(threshed_ref_db[k].max())
                thresh = threshed_ref_db[k].values[0]
                grow15 = np.clip(thresh * (1.15 ** (np.linspace(1, num_dates, num_dates, endpoint=True))), 0, max(max_value)).astype(int)
                fig.add_trace(go.Scatter(x=since_days, y=grow15, mode="lines", name='Grows 15% a day',
                                         line=dict(dash="dash", width=3, color=colors[cntry+1]), showlegend=showlegend),
                              row=1, col=cnt+1)  # threshed_ref_db[base]

                grow08 = np.clip(thresh * (1.08 ** (np.linspace(1, num_dates, num_dates, endpoint=True))), 0, max(max_value)).astype(int)
                fig.add_trace(go.Scatter(x=since_days, y=grow08, mode="lines", name='Grows 8% a day',
                                         line=dict(dash="dashdot", width=3, color=colors[cntry+2]), showlegend=showlegend),
                              row=1, col=cnt+1)

            if growth_rate_mean:
                cur_value = threshed_ref_db[k].values[-3]
                if cur_value > 0.8*max(max_value):
                    cur_value = min(max_value)
                grow_cur = np.clip(cur_value * (growth_rate_mean ** (np.linspace(1, num_dates, num_dates, endpoint=True))), 0, max(max_value)).astype(int)
                gr = int((growth_rate_mean - 1) * 100.0)
                fig.add_trace(go.Scatter(x=since_days, y=grow_cur, mode="lines",
                                         name='Grows ' + str(gr) + '% a day from last 3 days', showlegend=showlegend,
                                         line=dict(dash="dot", width=3, color=colors[cntry+3])), row=1, col=cnt+1)

    xaxis2 = 'Days since the ' + str(threshDays[1]) + 'th from the ' + str(threshValues[1]) + 'th case value'
    xaxis1 = 'Days since the ' + str(threshDays[0]) + 'th from the ' + str(threshValues[0]) + 'th case value'
    if log:
        fig.update_layout(hovermode="x", title=title_string, template='plotly_dark',
                          xaxis=dict(title=xaxis1), xaxis2=dict(title=xaxis2),
                          yaxis=dict(title=prefix + ' ' + inputs[0] + ' Cases', type='log'),
                          yaxis2=dict(title=prefix + ' ' + inputs[1] + ' Cases', type='log'))
    else:
        fig.update_layout(hovermode="x", title=title_string,  template='plotly_dark',
                      xaxis=dict(title=xaxis1), xaxis2=dict(title=xaxis2),
                      yaxis=dict(title=prefix + ' ' + inputs[0] + ' Cases'),
                      yaxis2=dict(title=prefix + ' ' + inputs[1] + ' Cases'))

    return fig
###################################################################################################################


# line plot
def line_country_plot(full_data, inputs=['Confirmed', 'Recovered', 'Deaths', 'Active'], base='Date', prefixes=[''],
                         fname=' Total Cases ', add_growth_rates=False, annotations=None, add_events_text=False,
                         factor=1.0, mat_plt=False, day=''):

    if not day:
        if isinstance(full_data.Date.max(), str):
            day = datetime.datetime.strptime(full_data.Date.max(), '%m/%d/%y').strftime('%d%m%y')
        else:
            day = full_data.Date.max().strftime('%d/%m/%y')

    try:
        not_country = 0
        country = full_data['Country'].unique()
        state = full_data['State'].unique()
    except:
        not_country = 1

    if not_country or country.shape[0] > 1:
        title_string = day + fname + 'Various Cases'
        save_string = full_data.Date.max().strftime('%d%m%y') + fname + '.png'
    elif state != country:
        title_string = country[0] + ' -- ' + state[0] + ' - ' + day + ' ' + fname
        save_string = full_data.Date.max().strftime('%d%m%y') + '_' + country[0] + '_' + state[0] + '_' +\
                      fname.replace(' ', '_') +'.png'
    else:
        title_string = state[0] + ' - ' + day + ' - ' + fname
        save_string = full_data.Date.max().strftime('%d%m%y') + '_' + state[0] + '_' + fname.replace(' ', '_') +'.png'

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Linear Plot", "Log Plot"))
    fig_cnt = -1
    customdata = None

    for pr_cnt in range(len(prefixes)):
        prefix = prefixes[pr_cnt]
        if prefix:
            colors = ['blue', 'yellow', 'green', 'magenta', 'cyan', 'red', 'black']
        else:
            colors = plotly.colors.DEFAULT_PLOTLY_COLORS
        for cnt in range(len(inputs)):
            case_k = inputs[cnt]
            k = prefix + case_k
            if k in full_data.keys():
                y = (full_data[k] * factor).fillna(0)
                # y[np.isinf(y)] = 0
                if base != 'Date':
                    customdata = full_data.Date
                if add_events_text:
                    trace = go.Scatter(x=full_data[base], y=y, mode="markers+lines+text", name=k, customdata=customdata,
                                       text=full_data.Event, marker=dict(size=4, color=colors[cnt]))
                else:
                    trace = go.Scatter(x=full_data[base], y=y, mode="markers+lines", name=k, customdata=customdata,
                                       marker=dict(size=4, color=colors[cnt]))
                fig.add_trace(trace, row=1, col=1)
                fig_cnt +=1
                fig.add_trace(trace, row=1, col=2)
                fig_cnt += 1
                if fig_cnt % 2 == 1:
                    fig.data[fig_cnt-1].update(showlegend=False)

            fig.update_traces(mode="markers+lines", hovertemplate=None)
            if base != 'Date':
                fig.update_traces(hovertemplate='%{y}<br>%{customdata| %_d %b %Y}')

            if add_growth_rates:
                grows_rate = full_data['Growth' + base].fillna(0).values / 100.0
                grows_rate[np.isinf(grows_rate)] = 0
                len_rate = len(grows_rate)
                vec = np.arange(0, round(len_rate*1/3))
                one_third = grows_rate[vec].mean()
                if one_third > 0:
                    grow_one_third = one_third * full_data[base] + full_data[k][vec[0]] * factor

                    add_trace1 = go.Scatter(x=full_data[base], y=grow_one_third, mode="lines",
                                            name='Linear estimation: ' + str(full_data[k][vec[0]]) + ' + '
                                                 + str(round(one_third, 2)) + '*' + base + '<br>' + str(round(one_third, 2))
                                            + ' - estim on first onethird of ' + base,
                                            line=dict(dash="dash", width=3))
                    fig.add_trace(add_trace1, row=1, col=1)
                    fig.add_trace(add_trace1, row=1, col=2)
                grows_rate = full_data['GrowthConfirmed'].fillna(0).values / 100.0
                grows_rate[np.isinf(grows_rate)] = 0
                len_rate = len(grows_rate)
                vec = np.arange(round(0.9*len_rate), len_rate)
                one_third = grows_rate[vec].mean()
                if one_third > 0:
                    grow_one_third = one_third * full_data[base] + full_data[k][vec[0]-round(0.1*len_rate)] * factor
                    add_trace2 = go.Scatter(x=full_data[base][round(len_rate*1/3):], y=grow_one_third[round(len_rate*1/3):],
                                          mode="lines", name='Linear estimation: '
                                                             + str(full_data[k][vec[0]-round(0.1*len_rate)]) + ' + '
                                                             + str(round(one_third, 2)) + '*' + base + '<br>'
                                                             + str(round(one_third, 2)) + ' - estim on 0.1 last from Confirmed',
                                            line=dict(dash="dash", width=3))
                    fig.add_trace(add_trace2, row=1, col=1)
                    fig.add_trace(add_trace2, row=1, col=2)
                fig.update_yaxes(range=[full_data[k][0], full_data[k][len_rate-1]], row=1, col=1)

    if annotations is not None:
        fig.update_annotations(annotations)

    fig.update_layout(template='plotly_dark', hovermode="x", title=title_string,
                      yaxis=dict(title=fname), xaxis=dict(title=base), yaxis2=dict(title=fname, type='log'),
                      xaxis2=dict(title=base))

    if mat_plt:
        fig_mat, ax = plt.subplots(figsize=(8, 6))
        colors = ['blue', 'green', 'yellow', 'magenta', 'cyan', 'red', 'black']
        max_values = []
        for cnt in range(len(inputs)):
            case_k = inputs[cnt]
            k = prefix + case_k
            full_data[k] = full_data[k].fillna(0)
            ax = sns.scatterplot(x=base, y=k, data=full_data, color=colors[cnt])
            plt.plot(full_data[base], full_data[k], zorder=1, color=colors[cnt], label=k)
            if not np.isinf(max(full_data[k])):
                max_values.append(max(full_data[k]))
        ax.set_xlim([full_data['Date'].iloc[0],  full_data['Date'].iloc[-1] + datetime.timedelta(days=1)])

        if max(full_data[prefix + inputs[0]]) > 1:
            max_value = max(max_values) + np.diff(full_data[k]).max()
            min_value = -1
        else:
            max_value = max(max_values) + np.diff(full_data[k]).max()
            min_value = 0
        ax.set_ylim([min_value, max_value])
        plt.legend(frameon=True, fontsize=12)
        plt.grid()
        plt.ylabel(fname)
        plt.title(title_string, fontsize=16)
        fig_mat.autofmt_xdate()
        plt.savefig(os.path.join(os.getcwd(), save_string))

    return fig
###################################################################################################################
