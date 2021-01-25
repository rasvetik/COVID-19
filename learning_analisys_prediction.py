"""
Created on Thursday Mar 26 2020
Sveta Raboy
based on
https://www.kaggle.com/bardor/covid-19-growing-rate
https://github.com/CSSEGISandData/COVID-19
https://github.com/imdevskp
https://www.kaggle.com/yamqwe/covid-19-status-israel
https://www.kaggle.com/vanshjatana/machine-learning-on-coronavirus
https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html
"""

import sys
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.neural_network import MLPRegressor
import time
import os
import numpy as np
import pandas as pd
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from Utils import *
from scipy.signal import argrelextrema

# seed ###################
seed = 1234
np.random.seed(seed)
##############################


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def calc_factor(db):
    # Factor to boost the calculation if the values are big
    if db.Confirmed.values[-1] > 1e8:
        factor = 100000.0
    elif db.Confirmed.values[-1] > 1e7:
        factor = 10000.0
    elif db.Confirmed.values[-1] > 1e6:
        factor = 1000.0
    elif db.Confirmed.values[-1] > 1e5:
        factor = 100.0
    elif db.Confirmed.values[-1] > 1e4:
        factor = 10.0
    else:
        factor = 1.0
    print('Boost factor %d' % factor)
    return factor


def local_extrema(in_data, do_smooth=True, window_len=15):
    if do_smooth:
        # moving average
        w = np.ones(window_len, 'd')
        data = np.convolve(w/w.sum(), in_data, mode='valid')
    else:
        data = in_data.values

    # for local maxima
    loc_maxima = argrelextrema(data, np.greater)[0]
    # for local minima
    loc_minima = argrelextrema(data, np.less)[0]

    return loc_maxima, loc_minima


def extend_index(date, new_size):
    values = date.values
    current = values[-1]
    while len(values) < new_size:
        current = current + np.timedelta64(1, 'D')
        values = np.append(values, current)
    return values


def extended_data(db, inputs, dates, prefix='Real'):

    size = len(dates)
    df = pd.DataFrame(index=np.datetime_as_string(dates, unit='D'))
    for cnt in range(len(inputs)):
        k = inputs[cnt]
        df[prefix + k] = np.concatenate((db[k].values, [None] * (size - len(db[k].values))))
    return df


def SIR_algo(data, predict_range=450, s_0=None, threshConfrirm=1, threshDays=None, active_ratio=0.11, debug_mode=None):
    # interactive site http://www.public.asu.edu/~hnesse/classes/sir.html
    # beta -  parameter controlling how much the disease can be transmitted through exposure.
    # gamma - parameter expressing how much the disease can be recovered in a specific period
    # r0 - basic reproduction number, the average number of people infected from one to other person betta/gamma
    # days - the average days to recover from infectious 1/gamma
    # epsilon- the D/R ratio describing whether overload of the health care system is approaching
    # delta_0 - learning cost of the system

    def loss(point, active, recovered, death,  s_0, i_0, r_0, d_0, alpha):
        # size = len(data)
        size = len(active)
        beta, gamma, delta = point

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            # return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
            return [-beta * S * I, beta * S * I - gamma * I - delta * D, gamma * I, delta * I]

        solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0, d_0], t_eval=np.arange(0, size, 1), vectorized=True)
        l1 = np.sqrt(np.mean((solution.y[1] - active) ** 2))
        l2 = np.sqrt(np.mean((solution.y[2] - recovered) ** 2))
        l3 = np.sqrt(np.mean((solution.y[3] - death) ** 2))

        return alpha[0] * l1 + np.max([0, 1 - alpha[0] - alpha[1] ]) * l2 + alpha[1] * l3

    def predict(dataDate, beta, gamma, delta, active, recovered, death, s_0, i_0, r_0, d_0):

        dates = extend_index(dataDate, predict_range)
        size = len(dates)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            # return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
            return [-beta * S * I, beta * S * I - gamma * I - delta * D, gamma * I, delta * I]

        prediction = solve_ivp(SIR, [0, size], [s_0, i_0, r_0, d_0], t_eval=np.arange(0, size, 1))
        pr_size = prediction.y.shape[1]
        if pr_size != size:
            new_size = pr_size - len(active.values)
            dates = dates[:pr_size]
        else:
            new_size = size - len(active.values)
        extended_active = np.concatenate((active.values, [None] * new_size))
        extended_recovered = np.concatenate((recovered.values, [None] * new_size))
        extended_death = np.concatenate((death.values, [None] * new_size))

        return dates, extended_active, extended_recovered, extended_death, prediction

    data = (data.loc[data.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    if threshDays:
        data = data.loc[:threshDays, :]
    cur_day = data.Date.max().strftime('%d%m%y')

    # Factor to boost the calculation if the values are big
    factor = calc_factor(data)

    Dsir = 0
    out_text = ''
    begin_idx = 0
    epsilon = []
    delta_0 = []
    max_id = data['Active'].idxmax()
    # idx_min_after_max = data['Active'][idx_max:].idxmin() + 1
    len_data = len(data['Active'])
    window_len = 15
    loc_max, loc_min = local_extrema(data['Active'], window_len=window_len)
    if len(loc_max) > len(loc_min):
        loc_min = np.append(loc_min, len_data + 1)
    elif len(loc_min) > len(loc_max):
        loc_max = np.append(loc_max, loc_max[-1])

    idx_max = loc_max[abs(loc_max - loc_min) > 10]
    idx_min_after_max = loc_min[abs(loc_max - loc_min) > 10]

    if len(idx_max) > 1:
        idx_max = np.unique(np.append((idx_max + (window_len - 1) / 2 - 1).astype(int), max_id))
        idx_min_after_max = np.unique(np.append((idx_min_after_max + (window_len - 1) / 2 - 1).astype(int), len_data + 1))
        idx_min_after_max = np.append(idx_min_after_max[0], idx_min_after_max[1:][np.diff(idx_min_after_max) > 10])
        wave = len(idx_min_after_max)
        print('There is exist ' + str(wave) + ' waves!')
    else:
        wave = 1
        idx_min_after_max = [len_data + 1]

    for cnt in range(wave):
        active_ratio = data['Active'][begin_idx:idx_min_after_max[cnt]].values[-1] / data['Confirmed'][begin_idx:idx_min_after_max[cnt]].values[-1]
        recovered = (data['Recovered'][begin_idx:idx_min_after_max[cnt]] / factor).reset_index().Recovered
        death = (data['Deaths'][begin_idx:idx_min_after_max[cnt]] / factor).reset_index().Deaths
        active = (data['Active'][begin_idx:idx_min_after_max[cnt]] / factor).reset_index().Active
        confirmed = (data['Confirmed'][begin_idx:idx_min_after_max[cnt]]).reset_index().Confirmed
        dataDate = data['Date'][begin_idx:idx_min_after_max[cnt]]
        try:
            country = data.Country.values[0]
        except:
            country = 'world'

        i_0 = active.values[0]
        r_0 = recovered.values[0]
        d_0 = death.values[0]
        if s_0 is None:
            s_0 = (confirmed.values[-1] / factor)
        else:
            s_0 = (s_0 / factor)

        alpha = [0.11, np.min([0.75, np.max([0.44, round(active_ratio, 3)])])]
        print('Suspected, WeightActive, WeightDeath')
        print([s_0, alpha])

        try:
            optimal = minimize(loss, [0.001, 0.001, 0.001], args=(active, recovered, death, s_0, i_0, r_0, d_0, alpha),
                               method='L-BFGS-B', bounds=[(0.00000001, 0.8), (0.00000001, 0.8), (0.00000001, 0.6)],
                               options={'maxls': 40, 'disp': debug_mode})
            print(optimal)
            if optimal.nit < 10 or ((round(1 / optimal.x[1]) < 13 or (1 / optimal.x[1]) > predict_range)
                                    and active_ratio > 0.075) or optimal.fun > 500:
                raise Exception('the parameters are not reliable')

        except Exception as exc:
            print(exc)
            try:
                optimal = minimize(loss, [0.001, 0.001, 0.001], args=(active, recovered, death, s_0, i_0, r_0, d_0, alpha),
                                   method='L-BFGS-B', bounds=[(0.00000001, 1), (0.00000001, 1), (0.00000001, 0.6)],
                                   options={'eps': 1e-7, 'maxls': 40, 'disp': debug_mode})
                print(optimal)
                if optimal.nit < 10 or ((round(1 / optimal.x[1]) < 14 or (1 / optimal.x[1]) > predict_range + 60)
                                        and active_ratio > 0.075) or optimal.fun > 600:
                    raise Exception('the parameters are not reliable')
            except Exception as exc:
                print(exc)
                optimal = minimize(loss, [0.01, 0.01, 0.01], args=(active, recovered, death, s_0, i_0, r_0, d_0, alpha),
                                   method='L-BFGS-B', bounds=[(0.00000001, 1), (0.00000001, 1), (0.00000001, 0.6)],
                                   options={'eps': 1e-5, 'maxls': 40, 'disp': debug_mode})
                print(optimal)
                if optimal.nit < 10 or ((round(1 / optimal.x[1]) < 15 or (1 / optimal.x[1]) > predict_range + 90)
                                        and active_ratio > 0.075) or optimal.fun > 700:
                    raise Exception('the parameters are not reliable')

        beta, gamma, delta = optimal.x
        dates, extended_active, extended_recovered, extended_death, prediction = \
            predict(dataDate, beta, gamma, delta, active, recovered, death, s_0, i_0, r_0, d_0)

        df = pd.DataFrame(
            {'Active Real': extended_active, 'Recovered Real': extended_recovered, 'Deaths Real': extended_death,
             # 'Susceptible': (prediction.y[0]).astype(int),
             'Active Predicted': (prediction.y[1]).astype(int),
             'Recovered Predicted': (prediction.y[2]).astype(int),
             'Deaths Predicted': (prediction.y[3]).astype(int)}, index=np.datetime_as_string(dates, unit='D'))

        df = df.mul(factor)
        df = df[df['Active Predicted'] >= 1]
        Dsir = Dsir + int((1 / gamma))
        dday = (data['Date'][idx_min_after_max[cnt]-2] + timedelta(1/gamma)).strftime('%d/%m/%y')
        # epsilon- the D/R ratio describing whether overload of the health care system is approaching
        epsilon.append(delta / gamma)
        # delta_0 - learning cost of the system
        delta_0.append(epsilon[cnt] * r_0 * factor - d_0 * factor)
        print('country=%s, wave=%d, beta=%.8f, gamma=%.8f, delta=%.8f, r_0=%.8f, d_0=%8.2f, epsilon=%.8f, days_to_recovery=%.1f'
              % (country, cnt+1, beta, gamma, delta, (beta / gamma), delta_0[cnt], epsilon[cnt], (1 / gamma)))

        if cnt == 0:
            full_data = df
            out_text = country + ' ' + str(data.Date.max().strftime('%d/%m/%y')) \
                               + ':  Since the ' + str(threshConfrirm) + ' Confirmed Case.'
            if wave > 1:
                full_data = df[: idx_min_after_max[cnt]]
                # begin_idx = idx_min_after_max + data['Active'][idx_min_after_max:].values.nonzero()[0][0]
                # idx_min_after_max = len_data + 1
                begin_idx = idx_min_after_max[cnt] + 1
                s_0 = None
                country_folder = os.path.join(os.getcwd(), time.strftime("%d%m%Y"), base_country)
                if not os.path.exists(country_folder):
                    os.makedirs(country_folder, exist_ok=True)
        else:
            df_text = country + ' ' + str(data.Date.max().strftime('%d/%m/%y')) \
                       + ':  Since the ' + str(threshConfrirm) + ' Confirmed Case in wave ' + str(cnt + 1) \
                       + '.  Days to recovery=' \
                       + str(Dsir) + ' - ' + str(dday) \
                       + '<br>\N{GREEK SMALL LETTER BETA}= ' + str(round(beta, 7)) \
                       + ',   \u03B3= ' + str(round(gamma, 7)) + ',   \u03B4= ' + str(round(delta, 7)) \
                       + ',   r\N{SUBSCRIPT ZERO}= ' + str(round((beta / gamma), 7)) \
                       + ',   \u03B4\N{SUBSCRIPT ZERO}= ' + str(round((delta_0[cnt]), 2)) \
                       + ',   \u03B5= ' + str(round((epsilon[cnt]), 7))

            fig, ax = plt.subplots(figsize=(14, 9))
            ax.set_title(df_text.replace('<br>', '\n'), loc='left')
            df.plot(ax=ax)
            save_string = cur_day + '_SIR_Prediction_' + country + 'wave '  + str(cnt + 1) + ' only' + '.png'
            fig.savefig(os.path.join(country_folder, save_string))
            if cnt == wave - 1:
                full_data = pd.concat([full_data, df], axis=0, sort=False)
            else:
                full_data = pd.concat([full_data, df[: (idx_min_after_max[cnt] - idx_min_after_max[cnt-1])]], axis=0, sort=False)
            begin_idx = idx_min_after_max[cnt] + 1
            s_0 = None

        out_text = out_text \
                   + '<br>Wave ' + str(cnt + 1) + ': Days to recovery=' + str(Dsir) + ' - ' + str(dday) \
                   + ', \N{GREEK SMALL LETTER BETA}= ' + str(round(beta, 7)) \
                   + ',   \u03B3= ' + str(round(gamma, 7)) + ',   \u03B4= ' + str(round(delta, 7)) \
                   + ',   r\N{SUBSCRIPT ZERO}= ' + str(round((beta / gamma), 7)) \
                   + ',   \u03B4\N{SUBSCRIPT ZERO}= ' + str(round((delta_0[cnt]), 2)) \
                   + ',   \u03B5= ' + str(round((epsilon[cnt]), 7))

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_title(out_text.replace('<br>', '\n'), loc='left')
        full_data.plot(ax=ax)
        plt.tight_layout()
        save_string = cur_day + '_SIR_Prediction_' + country + ' waves ' + str(cnt+1) + '.png'
        if wave > 1:
            fig.savefig(os.path.join(country_folder, save_string))
        if wave == cnt + 1:
            fig.savefig(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), save_string))

    return full_data, out_text, Dsir
##################################################################################################


def prophet_modeling_and_predicting(base_db, column_name, predict_range=365, first_n=45, last_n=30, threshConfrirm=1,
                                    threshDays=None, logistic=False, debug_mode=None):
    # Prophet Algorithm
    # Implements a procedure for forecasting time series data based on an additive model where non-linear trends are fit
    # with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong
    # seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend
    # and typically handles outliers well.
    data = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    if threshDays:
        data = data.loc[:threshDays, :]
    pr_data = data.loc[:, ['Date', column_name]].copy()
    pr_data.columns = ['ds', 'y']
    if logistic:
        growth = 'logistic'
        pr_data['cap'] = 2*pr_data.y.max()
    else:
        growth = 'linear'
    # Turn off fbprophet stdout logger
    logging.getLogger('Prophet').setLevel(logging.ERROR)
    # pr_data.y = pr_data.y.astype('float')
    # Modeling
    m = Prophet(growth=growth, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    with suppress_stdout_stderr():
        m.fit(pr_data)
    future = m.make_future_dataframe(periods=predict_range)
    if logistic:
        future['cap'] = 2 * pr_data.y.max()

    forecast_test = m.predict(future)

    # Predicting
    test = forecast_test.loc[:, ['ds', 'trend']]
    # test = test[test['trend'] > 0]
    test = test.head(first_n)
    forecast_test = forecast_test.head(first_n)
    if last_n < first_n:
        test = test.tail(last_n)
        forecast_test = forecast_test.tail(last_n)

    # Graphical Representation of Predicted Screening
    fig_test = plot_plotly(m, forecast_test, xlabel='Date', ylabel=column_name, trend=True)
    # fig_test.show()
    # # py.iplot(fig_test)  # only for Jupiter
    # f_test = m.plot(forecast_test, xlabel='Date', ylabel=column_name + ' Count')
    # figure_test = m.plot_components(forecast_test)
    test.columns = ['Date', column_name]

    return test, forecast_test, fig_test
##################################################################################################


def arima_modeling_and_predicting(base_db, column_name, predict_range=450, threshConfrirm=1, threshDays=None,
                                  debug_mode=None):
    # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    # Arima Algo - Autoregressive Integrated Moving Average Model
    # p -  auto-regressive aspect. this parameter says
    #                               it’s likely to rain tomorrow if it has been raining for the last 5 days
    # d - integrated part,  this parameter says
    #           it’s likely to rain the same amount tomorrow if the difference in rain in the last 5 days has been small
    # q - moving average part. this parameter sets the error of the model as a linear combination of the error values
    #                           observed at previous time points in the series
    # The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given
    # set of data. Lower - better
    data = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    # Factor to boost the calculation if the values are big
    factor = calc_factor(base_db)
    if threshDays:
        data = data.loc[:threshDays, :]
    arima_data = data.loc[:, ['Date', column_name]].copy()
    arima_data.columns = ['Date', 'Count']
    arima_data['Count'] = arima_data['Count'] / factor
    arima_data['Date'] = pd.to_datetime(arima_data['Date'])
    dates = extend_index(arima_data.Date, predict_range)
    size = len(dates)
    len_data = len(arima_data['Count'].values)
    period = size - len_data

    if debug_mode is not None:
        plt.figure()
        # Running the autocorrelation_plot, we can see where is a positive correlation (with the first PP lags)
        # and where that is perhaps significant for the first p lags (above the confidence line).
        autocorrelation_plot(arima_data['Count'])
        do_print = True
    else:
        do_print = False

    stepwise_fit = auto_arima(arima_data['Count'], d=2, D=2, trace=do_print,  # trace print log
                              error_action='ignore',  # we don't want to know if an order does not work
                              suppress_warnings=True,  # we don't want convergence warnings
                              stepwise=True)  # set to stepwise
    if do_print:
        # To print the summary
        print(stepwise_fit.summary())
        print('Arima order :' + str(stepwise_fit.order))

    order = tuple(np.array(stepwise_fit.order).clip(1, 3))
    model = ARIMA(arima_data['Count'].values, order=order)
    # Model and prediction
    # if stepwise_fit.order[0] == 0 or stepwise_fit.order[2] == 0:
    #     model = ARIMA(arima_data['Count'].values, order=(1, 2, 1))
    # else:
    #    model = ARIMA(arima_data['Count'].values, order=stepwise_fit.order)

    fit_model = model.fit(trend='c', full_output=True, disp=False)

    if do_print:
        print(fit_model.summary())
        fig, ax = plt.subplots(2, 2)
        # Graphical Representation for Prediction
        fit_model.plot_predict(ax=ax[0, 0])
        ax[0, 0].set_title('Forecast vs Actual for ' + column_name)
        # Plot residual errors
        residuals = pd.DataFrame(fit_model.resid)
        # if in the residual errors may still be some trend information not captured by the model.
        residuals.plot(title="Residual Error'", ax=ax[1, 0])
        # the density plot of the residual error values, suggesting the errors are Gaussian, but may not be centered on zero
        residuals.plot(kind='kde', title='Density', ax=ax[1, 1])

    # Forcast for next days (performs a one-step forecast using the model)
    forcast = fit_model.forecast(steps=period)
    pred_y = forcast[0].tolist()

    # Predictions of y values based on "model", namely fitted values
    # try:
    #     yhat = stepwise_fit.predict_in_sample(start=0, end=len_data-1)
    # except Exception as e:
    #     print(e)
    yhat = stepwise_fit.predict_in_sample()

    predictions = stepwise_fit.predict(period)
    # Calculate root mean squared error
    root_mse = rmse(arima_data['Count'], yhat)
    # Calculate mean squared error
    mse = mean_squared_error(arima_data['Count'], yhat)
    print('rmse=%d, mse=%d' % (root_mse, mse))
    if do_print:
        pd.DataFrame(pred_y).plot(title='Prediction, rmse=' + str(int(root_mse)), ax=ax[0, 1])

    test = pd.concat([pd.DataFrame(yhat, columns=[column_name]), pd.DataFrame(predictions, columns=[column_name])],
                     ignore_index=True)
    test = test.mul(factor)
    test.index = np.datetime_as_string(dates, unit='D')

    return test, root_mse
###########################################################################################################


def LSTM_modeling_and_predicting(base_db, column_name, predict_range=450, threshConfrirm=1,  threshDays=None,
                                 debug_mode=None):

    dataset = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    if threshDays:
        dataset = dataset.loc[:threshDays, :]
    data = dataset.loc[:, ['Date', column_name]].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    len_data = len(data[column_name].values)

    # Factor to boost the calculation if the values are big
    factor = calc_factor(base_db)

    y = data[column_name].values / factor

    # n_input observations will be used to predict the next value in the sequence
    n_input = np.max([8, round(3*len_data/4)])
    n_features = 1
    batch_size = np.min([8, len_data])
    predict_range = np.min([predict_range, len_data + n_input])

    # prepare data
    train_data = np.array(y).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)

    # prepare TimeSeriesGenerator
    generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=batch_size)

    if debug_mode is not None:
        # number of samples
        print('Samples: %d' % len(generator))
        do_print = True
        # print each sample
        see_generator = False
        if see_generator:
            for i in range(len(generator)):
                x, y = generator[i]
                print('%s => %s' % (x, y))
    else:
        do_print = False

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=48, return_sequences=True, input_shape=(n_input, n_features)))
    lstm_model.add(Dropout(0.2))
    # lstm_model.add(LSTM(units=64, return_sequences=True))
    # lstm_model.add(Dropout(0.1))
    lstm_model.add(LSTM(units=48))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    lstm_model.fit(generator, epochs=50, verbose=do_print)

    losses_lstm = lstm_model.history.history['loss']

    if do_print:
        plt.figure(figsize=(12, 4))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(np.arange(0, 100, 1))
        plt.plot(range(len(losses_lstm)), losses_lstm)

    lstm_predictions_scaled = []
    batch = scaled_train_data[:n_input]
    current_batch = batch.reshape((1, n_input, n_features))

    for i in range(predict_range-n_input):
        lstm_pred = lstm_model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[lstm_pred]], axis=1)

    prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled), columns=[column_name])
    prediction = pd.concat([pd.DataFrame(y[:n_input], columns=[column_name]), prediction], ignore_index=True)
    prediction = prediction.mul(factor)
    if do_print:
        prediction.plot(title='Prediction')
        data[column_name].plot(title=column_name)
        plt.title('Prediction of ' + column_name)
        plt.legend(['predicted ' + column_name, 'real ' + column_name])

    dates = extend_index(data.Date, predict_range)
    prediction.index = np.datetime_as_string(dates, unit='D')

    return prediction
######################################################################################################


def regression_modeling_and_predicting(base_db, column_name, predict_range=450, threshConfrirm=1, threshDays=None,
                                       debug_mode=None):
    # Class MLPRegressor implements a multi-layer perceptron (MLP)
    # that trains using backpropagation with no activation function in the output layer,
    # which can also be seen as using the identity function as activation function.
    # Therefore, it uses the square error as the loss function, and the output is a set of continuous values.

    base_db = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    if threshDays:
        base_db = base_db.loc[:threshDays, :]
    data = base_db.loc[:, ['Date', column_name]].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    len_data = len(data[column_name].values)
    # Factor to boost the calculation if the values are big
    factor = calc_factor(base_db)
    x = np.arange(len_data).reshape(-1, 1)
    y = data[column_name].values / factor

    model = MLPRegressor(hidden_layer_sizes=[64, 16], max_iter=100000, alpha=0.0001, random_state=11, verbose=False)

    # Regression Model
    do_scaling = False
    if do_scaling:
        # Model score is about -1 ---> the model can be arbitrarily worse for scaling option
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(x)
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y.reshape(-1, 1))
        model.fit(X_train, y_train)
    else:
        _ = model.fit(x, y)

    # Return the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    score = model.score(x, y)
    if debug_mode is not None:
        print(score)
        print(model.loss_)
        do_plt = True
    else:
        do_plt = False

    test = np.arange(predict_range).reshape(-1, 1)
    if do_scaling:
        p_pred = model.predict(scaler_x.fit_transform(test))
        pred = scaler_y.inverse_transform(p_pred.reshape(-1, 1))
    else:
        pred = model.predict(test)

    dates = [data['Date'][0] + timedelta(days=i) for i in range(predict_range)]
    dt_idx = pd.DatetimeIndex(dates)
    prediction = pred  # .round().astype(int)
    predicted_count = pd.DataFrame(prediction, columns=[column_name])
    predicted_count = predicted_count.mul(factor).round().astype(int)
    # Graphical representation of current confirmed and predicted confirmed
    if do_plt:
        accumulated_count = data[column_name]
        predicted_count.plot()
        accumulated_count.plot()
        plt.title('Prediction of Accumulated' + column_name + ' Count. Score=' + str(round(score, 2)))
        plt.legend(['predicted ' + column_name, 'real ' + column_name])

    predicted_count.index = np.datetime_as_string(dt_idx, unit='D')

    return predicted_count
#######################################################################################################


# If figures are not load in your default browser set Chrome as default
# print(plotly.io.renderers.default)
# plotly.io.renderers.default = 'chrome'

# Begin
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
some_countries = ['Israel', 'world', 'US', 'Russia', 'Brazil', 'Italy', 'Iran', 'Spain', 'France', 'Belgium',  'Sweden',
                  'Singapore', 'Switzerland', 'Turkey', 'Denmark', 'Germany', 'Austria', 'Australia', 'Japan', 'South Korea',
                  'Portugal', 'Norway', 'Qatar', 'Iceland', 'New Zealand', 'Panama', 'Estonia', 'Cyprus']

do_all = True
some = False
if do_all:
    countries = all_countries
elif some:
    countries = some_countries
else:
    countries = ['Israel', 'world']
# also the possibility for run if already some_countries were running or
# are not relevant due to absent some data
# Caution: In 'United Kingdom', 'Netherlands' the recovered data are absent
# remove_countries = ['United Kingdom', 'Netherlands', 'Ireland']
remove_countries = []
countries = [item for item in countries if item not in remove_countries]

# If prediction is not good the figure is shows (if flag activate) and not saved
show_not_good_prediction = False

stdoutOrigin = sys.stdout

for base_country in countries:
    # For Example Israel
    # base_country = 'Israel'  # may be: 'world' or country name like 'Russia' from the complete_data.csv
    fout = open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), 'learning_log.txt'), 'a')
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
        base_db = country_analysis(clean_db, world_population, country=base_country, state='', plt=True,
                                   fromFirstConfirm=True, num_days_for_rate=60)
        if base_country[-1] == '*':
            base_country = base_country[:-1]
            base_db['Country'] = base_country
        base_db.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), base_country + '_db.csv'), index=False)

    # range in days from threshold's Confirmed Cases
    predict_range = 450
    # threshold according to number of days. how many days use in estimation. default - all days till current day
    threshDays = None

    # SIR
    do_SIR = True
    # threshold on Confirmed value (from which value to begin estimation)
    # For SIR algo: Estimated percent of suspected population in %, for example 0.055%
    suspected_prcnt_pop = 0.05 / 100

    if do_SIR:
        print('Prediction with SIR Algorithm')
        try:
            data_db = base_db.copy()
            data_db['Date'] = pd.to_datetime(data_db['Date'])
            day = data_db.Date.max()
            # whether the number of suspected is estimated on % of population or according to current Confirmed value
            do_on_pop = False
            active_ratio = data_db.Active.values[-1] / data_db.Confirmed.values[-1]
            threshConfrirm = int(base_db.Confirmed.values[-1] * active_ratio * 0.01)
            # if active_ratio > 0.7:
            #     do_on_pop = True
            if do_on_pop:
                # there is estimation of 0.05% of population will be as suspected
                s_0 = np.max([data_db.Confirmed.values[-1], (data_db.Population.values[0]*suspected_prcnt_pop).astype(int)])
                print([threshConfrirm, data_db.Confirmed.values[-1],
                       (data_db.Population.values[0]*suspected_prcnt_pop).astype(int), round(active_ratio, 2)])
            else:
                s_0 = None
                print([threshConfrirm, data_db.Confirmed.values[-1], round(active_ratio, 2)])
            data, text, Dsir = SIR_algo(data_db, predict_range=predict_range, s_0=s_0, threshConfrirm=threshConfrirm,
                                        active_ratio=active_ratio)
            sir_annot = dict(xref='paper', yref='paper', x=0.35, y=0.93, align='left', font=dict(size=12), text=text)
            data['Date'] = data.index
            data.Date = pd.to_datetime(data.Date)
            with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
                fsc1 = scatter_country_plot(data, fname=' - ' + base_country + ' - Prediction with SIR Algorithm ',
                                            inputs=data.keys()[:-1], annotations=sir_annot, day=day.strftime('%d/%m/%y'))
                f.write(fsc1.to_html(full_html=False, include_plotlyjs='cdn'))

        except Exception as e:
            print(e)
            try:
                print('Last Try')
                threshConfrirm = 1
                data, text, Dsir = SIR_algo(data_db, predict_range=predict_range, s_0=s_0,
                                            threshConfrirm=threshConfrirm, active_ratio=active_ratio)
                sir_annot = dict(xref='paper', yref='paper', x=0.25, y=0.95, align='left', font=dict(size=14),
                                 text=text)
                data['Date'] = data.index
                data.Date = pd.to_datetime(data.Date)
                with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"),
                                       day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
                    fsc1 = scatter_country_plot(data, fname=' - ' + base_country + ' - Prediction with SIR Algorithm ',
                                                inputs=data.keys()[:-1], annotations=sir_annot,
                                                day=day.strftime('%d/%m/%y'))
                    f.write(fsc1.to_html(full_html=False, include_plotlyjs='cdn'))

            except Exception as e:
                print(e)
                print('Not executed Prediction with SIR Algorithm. May be some data are absent. '
                      'May be were some problems in data reporting')
    fout.close()
    # sys.stdout.close()
    sys.stdout = stdoutOrigin

    ##################################################################################################################
    # Prophet Algorithm
    do_prophet = True
    # threshold on Confirmed value (from which value to begin estimation)
    threshConfrirm = 1
    if do_prophet:
        print('Prediction with Prophet Algorithm')
        try:
            # For Confirmed Cases
            cnfrm, forecast_cnfrm, fig_cnfrm = prophet_modeling_and_predicting(base_db, 'Confirmed', predict_range=predict_range,
                                                first_n=predict_range, last_n=predict_range, threshConfrirm=threshConfrirm)
            # For Recover Cases
            # This prediction is truly based on the dataset depend on the current situation.
            # In future if we able to get vaccine there will be gradual changes in recovery
            rec, forecast_rec, fig_rec = prophet_modeling_and_predicting(base_db, 'Recovered', predict_range=predict_range,
                                                first_n=predict_range, last_n=predict_range, threshConfrirm=threshConfrirm)
            # For Death
            dth, forecast_dth, fig_dth = prophet_modeling_and_predicting(base_db, 'Deaths', predict_range=predict_range,
                                                first_n=predict_range, last_n=predict_range, threshConfrirm=threshConfrirm)

            # For Active - not usable because of linear or logistic nature of predictions only
            # act, forecast_act, fig_act = prophet_modeling_and_predicting(base_db, 'Active', predict_range=predict_range,
            #                                    first_n=predict_range, last_n=predict_range, threshConfrirm=threshConfrirm)

            # How future looks like!!
            data = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
            len_data = data.shape[0]
            data['Date'] = pd.to_datetime(data['Date'])
            day = data.Date.max()
            dates = extend_index(data.Date, predict_range)
            inputs = ['Confirmed', 'Deaths', 'Recovered', 'Active']
            prop_df = extended_data(data, inputs, dates, prefix='')
            size = len(dates)

            prop_df['PredictedConfirmed'] = cnfrm.Confirmed.values.clip(0).astype(int)
            prop_df['PredictedRecovered'] = rec.Recovered.values.clip(0).astype(int)
            prop_df['PredictedDeaths'] = dth.Deaths.values.clip(0).astype(int)
            # prop_df['PredictedActive'] = act.Active.values.clip(0).astype(int)
            prop_df['LowPredictedConfirmed'] = forecast_cnfrm.trend_lower.values.clip(0).astype(int)
            prop_df['LowPredictedRecovered'] = forecast_rec.trend_lower.values.clip(0).astype(int)
            prop_df['LowPredictedDeaths'] = forecast_dth.trend_lower.values.clip(0).astype(int)
            # prop_df['LowPredictedActive'] = forecast_act.yhat_lower.values.clip(0).astype(int)
            prop_df['HighPredictedConfirmed'] = forecast_cnfrm.trend_upper.values.clip(0).astype(int)
            prop_df['HighPredictedRecovered'] = forecast_rec.trend_upper.values.clip(0).astype(int)
            prop_df['HighPredictedDeaths'] = forecast_dth.trend_upper.values.clip(0).astype(int)
            # prop_df['HighPredictedActive'] = forecast_act.trend_upper.values.clip(0).astype(int)
            prop_df = prop_df.fillna(0)
            # It is not considered the death values for more relax date recovery value
            prop_df = prop_df[prop_df['PredictedRecovered'] <= prop_df['PredictedConfirmed']]
            prop_df['Date'] = prop_df.index
            prop_df.Date = pd.to_datetime(prop_df.Date)

            Dprop = prop_df['Date'].max() - day

            # Future Ratio and percentages
            pr_pps = float(prop_df.PredictedRecovered[-1]/prop_df.PredictedConfirmed[-1])
            pd_pps = float(prop_df.PredictedDeaths[-1]/prop_df.PredictedConfirmed[-1])

            print("The percentage of Predicted recovery after confirmation is " + str(round(pr_pps*100, 2)))
            print("The percentage of Predicted Death after confirmation is " + str(round(pd_pps*100, 2)))
            print('Days to recovery ' + str(Dprop.days) + ' - ' + prop_df['Date'].max().strftime('%d/%m/%y'))

            pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.95, align='left', font=dict(size=14),
                            text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                                 + '<br>Low Bound Predicted Recovery after confirmation is ' + str(round(pr_pps*100, 1))
                                 + '%<br>Low Bound Predicted Death after confirmation is ' + str(round(pd_pps*100, 1)) + '%'
                                 + '<br>Days to recovery ' + str(Dprop.days) + ' - ' + prop_df['Date'].max().strftime('%d/%m/%y'))

            with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
                fsc2 = line_country_plot(prop_df,
                                         fname=' - ' + base_country + ' - Prediction with Prophet Algorithm ',
                                         prefixes=['', 'Predicted', 'LowPredicted', 'HighPredicted'],
                                         annotations=pred_ann, day=day.strftime('%d/%m/%y'))
                if pr_pps + pd_pps < 0.75:
                    # If prediction is not good the figure is shows only and not saved
                    if show_not_good_prediction:
                        fsc2.show()
                    pass
                else:
                    f.write(fsc2.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            print(e)
            print('Not executed Prediction with Prophet Algorithm')


    ###############################################################################################################
    # Arima Algo - Autoregressive Integrated Moving Average Model
    do_ARIMA = True
    threshConfrirm = int(base_db.Confirmed.values[-1] * 0.01)
    if do_ARIMA:
        print('Prediction with ARIMA Algorithm')
        try:
            # For Confirmed Cases
            acnfrm, mse_cnfrm = arima_modeling_and_predicting(base_db, 'Confirmed', predict_range=predict_range,
                                                              threshConfrirm=threshConfrirm)
            # For Recover Cases
            # This prediction is truly based on the dataset depend on the current situation.
            # In future if we able to get vaccine there will be gradual changes in recovery
            arec, mse_rec = arima_modeling_and_predicting(base_db, 'Recovered', predict_range=predict_range,
                                                          threshConfrirm=threshConfrirm)
            # For Death
            adth, mse_dth = arima_modeling_and_predicting(base_db, 'Deaths', predict_range=predict_range,
                                                          threshConfrirm=threshConfrirm)
            # For Active
            aact, mse_act = arima_modeling_and_predicting(base_db, 'Active', predict_range=predict_range,
                                                          threshConfrirm=threshConfrirm)

            # How future looks like!!
            data = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
            len_data = data.shape[0]
            data['Date'] = pd.to_datetime(data['Date'])
            day = data.Date.max()

            dates = extend_index(data.Date, predict_range)
            inputs = ['Confirmed', 'Deaths', 'Recovered', 'Active']
            aprop_df = extended_data(data, inputs, dates)
            size = len(dates)

            aprop_df['PredictedConfirmed'] = acnfrm.Confirmed.values.astype(int)
            aprop_df['PredictedRecovered'] = arec.Recovered.values.astype(int)
            aprop_df['PredictedDeaths'] = adth.Deaths.values.astype(int)
            aprop_df['PredictedActive'] = aact.Active.values.astype(int)
            aprop_df = aprop_df.fillna(0)
            # It is not considered the death values for more relax date recovery value
            aprop_df = aprop_df[aprop_df['PredictedRecovered'] <= aprop_df['PredictedConfirmed']]
            aprop_df = aprop_df[aprop_df['PredictedActive'] >= 0]
            aprop_df['Date'] = aprop_df.index
            aprop_df.Date = pd.to_datetime(aprop_df.Date)

            # Future Ratio and percentages
            apr_pps = float(aprop_df.PredictedRecovered[-1]/aprop_df.PredictedConfirmed[-1])
            apd_pps = float(aprop_df.PredictedDeaths[-1]/aprop_df.PredictedConfirmed[-1])
            Daprop = aprop_df['Date'].max() - day

            print("The percentage of Predicted recovery after confirmation is %.2f" % (apr_pps*100))
            print("The percentage of Predicted Death after confirmation is %.2f" % (apd_pps*100))
            print('Days to recovery=' + str(Daprop.days) + ' - ' + aprop_df['Date'].max().strftime('%d/%m/%y'))

            pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.9, align='left', font=dict(size=14),
                            text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                                 + '<br>rmse(Cnfrm)=' + str(int(mse_cnfrm)) + ', rmse(Recv)=' + str(int(mse_rec))
                                 + ', rmse(Dth)=' + str(int(mse_dth)) + ', rmse(Actv)=' + str(int(mse_act))
                                 + '<br>Predicted Recovery after confirmation is ' + str(round(apr_pps*100, 1))
                                 + '%<br>Predicted Death after confirmation is ' + str(round(apd_pps*100, 1))
                                 + '<br>Days to recovery ' + str(Daprop.days) + ' - ' + aprop_df['Date'].max().strftime('%d/%m/%y'))

            with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
                fsc3 = scatter_country_plot(aprop_df, fname=' - ' + base_country + ' - Prediction with ARIMA Algorithm ',
                                            inputs=aprop_df.keys()[:-1], annotations=pred_ann, day=day.strftime('%d/%m/%y'))
                if apr_pps + apd_pps < 0.75:
                    # If prediction is not good the figure is shows only and not saved
                    if show_not_good_prediction:
                        fsc3.show()
                    pass
                else:
                    f.write(fsc3.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            print(e)
            print('Not executed Prediction with ARIMA Algorithm')

    #########################################################################################
    # LSTM - take a time to solve
    do_LSTM = True
    # threshold on Confirmed value (from which value to begin estimation)
    threshConfrirm = int(base_db.Confirmed.values[-1] * 0.001)
    if do_LSTM:
        print('Prediction with LSTM Algorithm')
        try:
            lstm_active = LSTM_modeling_and_predicting(base_db, 'Active', predict_range=predict_range, threshConfrirm=threshConfrirm)
            lstm_cnfrm = LSTM_modeling_and_predicting(base_db, 'Confirmed', predict_range=predict_range, threshConfrirm=threshConfrirm)
            lstm_rec = LSTM_modeling_and_predicting(base_db, 'Recovered', predict_range=predict_range, threshConfrirm=threshConfrirm)
            lstm_dth = LSTM_modeling_and_predicting(base_db, 'Deaths', predict_range=predict_range, threshConfrirm=threshConfrirm)

            # How future looks like!!
            data = base_db.copy()
            data = (data.loc[data.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
            len_data = data.shape[0]
            data['Date'] = pd.to_datetime(data['Date'])
            day = data.Date.max()
            cur_predict_range = np.min([predict_range, lstm_active.shape[0]])
            dates = extend_index(data.Date, cur_predict_range)
            inputs = ['Confirmed', 'Deaths', 'Recovered', 'Active']
            lstm_prop_df = extended_data(data, inputs, dates)
            size = len(dates)

            lstm_prop_df['PredictedConfirmed'] = lstm_cnfrm.Confirmed.astype(int)
            lstm_prop_df['PredictedRecovered'] = lstm_rec.Recovered.astype(int)
            lstm_prop_df['PredictedDeaths'] = lstm_dth.Deaths.astype(int)
            lstm_prop_df['PredictedActive'] = lstm_active.Active.astype(int)

            lstm_prop_df = lstm_prop_df.fillna(0)

            lstm_prop_df['Date'] = lstm_prop_df.index
            lstm_prop_df.Date = pd.to_datetime(lstm_prop_df.Date)
            # It is not considered the death values for more relax date recovery value
            lstm_prop_df = lstm_prop_df[lstm_prop_df['PredictedRecovered'] <= lstm_prop_df['PredictedConfirmed']]
            lstm_prop_df = lstm_prop_df.reset_index()
            lstm_prop_df.pop('index')

            idx_real = lstm_prop_df.RealActive.to_numpy().nonzero()[0][-1]
            idx_pred = np.where((lstm_prop_df.PredictedActive[idx_real:] == lstm_prop_df.PredictedActive.values[-1]))[0][0]

            idx = idx_real + idx_pred
            if idx >= idx_real:
                lstm_prop_df = lstm_prop_df.loc[:idx]

            # Future Ratio and percentages
            lpr_pps = float(lstm_prop_df.PredictedRecovered.values[-1]/lstm_prop_df.PredictedConfirmed.values[-1])
            lpd_pps = float(lstm_prop_df.PredictedDeaths.values[-1]/lstm_prop_df.PredictedConfirmed.values[-1])

            Dlstm = lstm_prop_df['Date'].max() - day

            print("The percentage of Predicted recovery after confirmation is %.2f" % (lpr_pps*100))
            print("The percentage of Predicted Death after confirmation is %.2f" % (lpd_pps*100))
            print('Days to recovery ' + str(Dlstm.days) + ' - ' + lstm_prop_df['Date'].max().strftime('%d/%m/%y'))

            pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.9, align='left', font=dict(size=14),
                            text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                                 + '<br>Predicted Recovery after confirmation is ' + str(round(lpr_pps*100, 1))
                                 + '%<br>Predicted Death after confirmation is ' + str(round(lpd_pps*100, 1))
                                 + '<br>Days to recovery ' + str(Dlstm.days) + ' - '
                                 + lstm_prop_df['Date'].max().strftime('%d/%m/%y'))

            with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
                fsc4 = scatter_country_plot(lstm_prop_df,
                                            fname=' - ' + base_country + ' - Prediction with LSTM Algorithm ',
                                            inputs=lstm_prop_df.keys()[:-1], annotations=pred_ann, day=day.strftime('%d/%m/%y'))
                if lpr_pps + lpd_pps < 0.55:
                    # If prediction is not good the figure is shows only and not saved
                    if show_not_good_prediction:
                        fsc4.show()
                    pass
                else:
                    f.write(fsc4.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            print(e)
            print('Not executed Prediction with LSTM Algorithm')

    ################################################################################################
    # Regression
    do_reg = True
    threshConfrirm = 1
    if do_reg:
        print('Prediction with Multi-layer Perceptron Regressor Algorithm')
        try:
            reg_cnfrm = regression_modeling_and_predicting(base_db, 'Confirmed', predict_range=predict_range, threshConfrirm=threshConfrirm)
            reg_rec = regression_modeling_and_predicting(base_db, 'Recovered', predict_range=predict_range, threshConfrirm=threshConfrirm)
            reg_dth = regression_modeling_and_predicting(base_db, 'Deaths', predict_range=predict_range, threshConfrirm=threshConfrirm)
            reg_act = regression_modeling_and_predicting(base_db, 'Active', predict_range=predict_range, threshConfrirm=threshConfrirm)

            # How future looks like!!
            data = base_db.copy()
            data = (data.loc[data.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
            data['Date'] = pd.to_datetime(data['Date'])
            dates = extend_index(data.Date, predict_range)
            inputs = ['Confirmed', 'Deaths', 'Recovered', 'Active']
            reg_prop_df = extended_data(data, inputs, dates)
            size = len(dates)

            reg_prop_df['PredictedConfirmed'] = reg_cnfrm.Confirmed
            reg_prop_df['PredictedRecovered'] = reg_rec.Recovered
            reg_prop_df['PredictedDeaths'] = reg_dth.Deaths
            reg_prop_df['PredictedActive'] = reg_act.Active
            reg_prop_df['PredictedConfirmed'] = reg_act.Active + reg_dth.Deaths + reg_rec.Recovered
            reg_prop_df = reg_prop_df.fillna(0)

            day = data.Date.max()
            reg_prop_df['Date'] = reg_prop_df.index
            reg_prop_df.Date = pd.to_datetime(reg_prop_df.Date)
            # It is not considered the death values for more relax date recovery value
            reg_prop_df = reg_prop_df[reg_prop_df['PredictedRecovered'] <= reg_prop_df['PredictedConfirmed']]
            reg_prop_df = reg_prop_df[reg_prop_df['PredictedActive'] > 0]
            reg_prop_df = reg_prop_df.reset_index()
            reg_prop_df.pop('index')

            idx_real = reg_prop_df.RealActive.to_numpy().nonzero()[0][-1]
            # idx_pred = reg_prop_df.PredictedActive[reg_prop_df.PredictedActive[idx_real:] ==
            #                                        reg_prop_df.PredictedActive.values[-1]].index[0]
            idx_pred = np.where((reg_prop_df.PredictedActive[idx_real:] == reg_prop_df.PredictedActive.values[-1]))[0][0]
            idx = idx_real + idx_pred
            if idx >= idx_real:
                reg_prop_df = reg_prop_df.loc[:idx]
            # Future Ratio and percentages
            rpr_pps = float(reg_prop_df.PredictedRecovered.values[-1]/reg_prop_df.PredictedConfirmed.values[-1])
            rpd_pps = float(reg_prop_df.PredictedDeaths.values[-1]/reg_prop_df.PredictedConfirmed.values[-1])

            Dreg = reg_prop_df['Date'].max() - day

            print("The percentage of Predicted recovery after confirmation is %.2f" % (rpr_pps*100))
            print("The percentage of Predicted Death after confirmation is %.2f" % (rpd_pps*100))
            print('Days to recovery ' + str(Dreg.days) + ' - ' + reg_prop_df['Date'].max().strftime('%d/%m/%y'))

            pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.9, align='left', font=dict(size=14),
                            text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                                 + '<br>Predicted Recovery after confirmation is ' + str(round(rpr_pps*100, 1))
                                 + '%<br>Predicted Death after confirmation is ' + str(round(rpd_pps*100, 1))
                                 + '<br>Days to recovery ' + str(Dreg.days) + ' - '
                                 + reg_prop_df['Date'].max().strftime('%d/%m/%y'))

            with open(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
                fsc5 = scatter_country_plot(reg_prop_df,
                                            fname=' - ' + base_country + ' - Prediction with Multi-layer Perceptron Regressor Algorithm ',
                                            inputs=reg_prop_df.keys()[:-1], annotations=pred_ann, day=day.strftime('%d/%m/%y'))
                if rpr_pps + rpd_pps < 0.75:
                    # If prediction is not good the figure is shows only and not saved
                    if show_not_good_prediction:
                        fsc5.show()
                    pass
                else:
                    f.write(fsc5.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            print(e)
            print('Not executed Prediction with Multi-layer Perceptron Regressor Algorithm')

    plt.close('all')

