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

import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.neural_network import MLPRegressor
import time
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from Utils import *


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


def SIR_algo(data, predict_range=150, s_0=None, threshConfrirm=1):
    # beta -  parameter controlling how much the disease can be transmitted through exposure.
    # gamma - parameter expressing how much the disease can be recovered in a specific period
    # r0 - basic reproduction number.the average number of people infected from one other person betta/gamma
    # days - the average days to recover from infectious 1/gamma

    def loss(point, active, recovered, death,  s_0, i_0, r_0, d_0, alpha):
        size = len(data)
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

        return alpha[0] * l1 + (1 - alpha[0] - alpha[1]) * l2 + alpha[1] * l3

    def predict(data, beta, gamma, delta, active, recovered, death, s_0, i_0, r_0, d_0):

        dates = extend_index(data.Date, predict_range)
        size = len(dates)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            # return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
            return [-beta * S * I, beta * S * I - gamma * I - delta * D, gamma * I, delta * I]

        extended_active = np.concatenate((active.values, [None] * (size - len(active.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        prediction = solve_ivp(SIR, [0, size], [s_0, i_0, r_0, d_0], t_eval=np.arange(0, size, 1))

        return dates, extended_active, extended_recovered, extended_death, prediction

    data = (data.loc[data.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    cur_day = data.Date.max().strftime('%d%m%y')
    recovered = data.Recovered
    death = data.Deaths
    active = data.Active
    try:
        country = data.Country.values[0]
    except:
        country = ''

    i_0 = active.values[0]
    r_0 = recovered.values[0]
    d_0 = death.values[0]
    if s_0 is None:
        s_0 = data.Confirmed.values[-1]

    alpha = [0.11, 0.225]
    print('Suspected, WeightActive, WeightDeath')
    print([s_0, alpha])

    optimal = minimize(loss, [0.001, 0.001, 0.001], args=(active, recovered, death, s_0, i_0, r_0, d_0, alpha),
                       method='L-BFGS-B', bounds=[(0.00000001, 0.1), (0.00000001, 0.1), (0.00000001, 0.1)])
    print(optimal)
    beta, gamma, delta = optimal.x
    dates, extended_active, extended_recovered, extended_death, prediction = \
        predict(data, beta, gamma, delta, active, recovered, death, s_0, i_0, r_0, d_0)

    df = pd.DataFrame(
        {'Active Real': extended_active, 'Recovered Real': extended_recovered, 'Deaths Real': extended_death,
         'Susceptible': prediction.y[0].astype(int), 'Active Predicted': prediction.y[1].astype(int),
         'Recovered Predicted': prediction.y[2].astype(int),
         'Deaths Predicted': prediction.y[3].astype(int)}, index=np.datetime_as_string(dates, unit='D'))

    df = df[df['Active Predicted'] >= 0]
    Dsir = int((1 / gamma))
    dday = (data.Date.max() + timedelta(1/gamma)).strftime('%d/%m/%y')
    out_text = country + ':  Since the ' + str(threshConfrirm) + ' Confirmed Case.  Days to recovery='\
                       + str(Dsir) + ' - ' + str(dday) \
                       + '<br>\N{GREEK SMALL LETTER BETA}= ' + str(round(beta, 7)) \
                       + ',   \u03B3= ' + str(round(gamma, 7)) + ',   \u03B4= ' + str(round(delta, 7)) \
                       + ',   r\N{SUBSCRIPT ZERO}= ' + str(round((beta / gamma), 7))

    print('country=%s, beta=%.8f, gamma=%.8f, delta=%.8f, r_0=%.8f, days_to_recovery=%.1f'
          % (country, beta, gamma, delta, (beta / gamma), (1 / gamma)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(country)
    df.plot(ax=ax)

    save_string = cur_day + '_SIR_Prediction_' + country + '.png'
    fig.savefig(os.path.join(os.getcwd(), save_string))

    return df, out_text, Dsir


def prophet_modeling_and_predicting(base_db, column_name, predict_range=365, first_n=45, last_n=30, threshConfrirm=1):
    # Prophet Algorithm
    # Implements a procedure for forecasting time series data based on an additive model where non-linear trends are fit
    # with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong
    # seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend
    # and typically handles outliers well.
    data = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    pr_data = data.loc[:, ['Date', column_name]].copy()
    pr_data.columns = ['ds', 'y']

    # Modeling
    m = Prophet()
    m.fit(pr_data)
    future = m.make_future_dataframe(periods=predict_range)
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


def arima_modeling_and_predicting(base_db, column_name, predict_range=60, threshConfrirm=1):
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
    arima_data = data.loc[:, ['Date', column_name]].copy()
    arima_data.columns = ['Date', 'Count']
    arima_data['Date'] = pd.to_datetime(arima_data['Date'])
    dates = extend_index(arima_data.Date, predict_range)
    size = len(dates)
    len_data = len(arima_data['Count'].values)
    period = size - len_data
    plt.figure()
    autocorrelation_plot(arima_data['Count'])

    stepwise_fit = auto_arima(arima_data['Count'], d=2, D=2, trace=True,
                              error_action='ignore',  # we don't want to know if an order does not work
                              suppress_warnings=True,  # we don't want convergence warnings
                              stepwise=True)  # set to stepwise
    # To print the summary
    print(stepwise_fit.summary())

    # Model and prediction
    model = ARIMA(arima_data['Count'].values, order=stepwise_fit.order)  # order=(1, 2, 1)
    fit_model = model.fit(trend='c', full_output=True, disp=True)
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
    yhat = stepwise_fit.predict_in_sample(start=0, end=len_data-1)
    predictions = stepwise_fit.predict(period)
    # Calculate root mean squared error
    root_mse = rmse(arima_data['Count'], yhat)
    # Calculate mean squared error
    mse = mean_squared_error(arima_data['Count'], yhat)
    print('rmse=%d, mse=%d' % (root_mse, mse))
    pd.DataFrame(pred_y).plot(title='Prediction, rmse=' + str(int(root_mse)), ax=ax[0, 1])

    test = pd.concat([pd.DataFrame(yhat, columns=[column_name]), pd.DataFrame(predictions, columns=[column_name])],
                     ignore_index=True)
    test.index = np.datetime_as_string(dates, unit='D')

    return test, root_mse
###########################################################################################################


def LSTM_modeling_and_predicting(base_db, column_name, predict_range=150, threshConfrirm=1):

    dataset = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    data = dataset.loc[:, ['Date', column_name]].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    dates = extend_index(data.Date, predict_range)
    len_data = len(data[column_name].values)

    y = data[column_name].values

    # n_input observations will be used to predict the next value in the sequence
    n_input = np.max([8, round(3*len_data/4)])
    n_features = 1
    batch_size = np.min([8, len_data])

    # prepare data
    train_data = np.array(y).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)

    # prepare TimeSeriesGenerator
    generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=batch_size)
    # number of samples
    print('Samples: %d' % len(generator))
    # print each sample
    do_print = False
    if do_print:
        for i in range(len(generator)):
            x, y = generator[i]
            print('%s => %s' % (x, y))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=48, return_sequences=True, input_shape=(n_input, n_features)))
    lstm_model.add(Dropout(0.2))
    # lstm_model.add(LSTM(units=64, return_sequences=True))
    # lstm_model.add(Dropout(0.1))
    lstm_model.add(LSTM(units=48))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    lstm_model.fit(generator, epochs=100)

    losses_lstm = lstm_model.history.history['loss']
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
    prediction.plot(title='Prediction')
    data[column_name].plot(title=column_name)
    plt.title('Prediction of ' + column_name)
    plt.legend(['predicted ' + column_name, 'real ' + column_name])

    prediction.index = np.datetime_as_string(dates, unit='D')

    return prediction
######################################################################################################


def regression_modeling_and_predicting(base_db, column_name, predict_range=150, threshConfrirm=1):
    # Class MLPRegressor implements a multi-layer perceptron (MLP)
    # that trains using backpropagation with no activation function in the output layer,
    # which can also be seen as using the identity function as activation function.
    # Therefore, it uses the square error as the loss function, and the output is a set of continuous values.

    base_db = (base_db.loc[base_db.loc[:, 'Confirmed'] > threshConfrirm, :]).reset_index()
    data = base_db.loc[:, ['Date', column_name]].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    x = np.arange(len(data)).reshape(-1, 1)
    y = data[column_name].values

    model = MLPRegressor(hidden_layer_sizes=[32, 32, 8], max_iter=50000, alpha=0.0001, random_state=11, verbose=True)

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
    print(score)
    print(model.loss_)

    test = np.arange(predict_range).reshape(-1, 1)
    if do_scaling:
        p_pred = model.predict(scaler_x.fit_transform(test))
        pred = scaler_y.inverse_transform(p_pred.reshape(-1, 1))
    else:
        pred = model.predict(test)

    prediction = pred.round().astype(int)
    dates = [data['Date'][0] + timedelta(days=i) for i in range(predict_range)]
    dt_idx = pd.DatetimeIndex(dates)
    predicted_count = pd.DataFrame(prediction, columns=[column_name])

    # Graphical representation of current confirmed and predicted confirmed
    accumulated_count = data[column_name]
    predicted_count.plot()
    accumulated_count.plot()
    plt.title('Prediction of Accumulated' + column_name + ' Count. Score=' + str(round(score, 2)))
    plt.legend(['predicted ' + column_name, 'real ' + column_name])

    predicted_count.index = np.datetime_as_string(dt_idx, unit='D')

    return predicted_count
#######################################################################################################


# Begin
full_data_file = os.path.join(os.getcwd(), time.strftime("%d%m%Y") + 'complete_data.csv')
world_pop_file = os.path.join(os.getcwd(), 'world_population_csv.csv')

# Israel
base_country = 'Israel'
base_country_file = os.path.join(os.getcwd(), base_country + '_db.csv')

if os.path.exists(base_country_file):
    base_db = pd.read_csv(base_country_file)

elif os.path.exists(full_data_file):
    clean_db = pd.read_csv(full_data_file)
    world_population = pd.read_csv(world_pop_file)
    clean_db['Date'] = pd.to_datetime(clean_db['Date'])

    # Sort by Date
    daily = clean_db.sort_values(['Date', 'Country', 'State'])
    base_db = country_analysis(clean_db, world_population, country=base_country, state='', plt=True, fromFirstConfirm=True)
    base_db.to_csv(os.path.join(os.getcwd(), base_country + '_db.csv'), index=False)


threshConfrirm = int(base_db.Confirmed.values[-1] * 0.01)
# range in days from threshold's Confirmed Cases
predict_range = 365
do_on_pop = False

##################################################################################################
# SIR
do_SIR = True
if do_SIR:
    try:
        data_db = base_db.copy()
        data_db['Date'] = pd.to_datetime(data_db['Date'])
        day = data_db.Date.max()
        if do_on_pop:
            # there is estimation of 0.1% of population will be as suspected
            s_0 = np.max([data_db.Confirmed.values[-1], (data_db.Population.values[0]*0.001).astype(int)])
        else:
            s_0 = None
        data, text, Dsir = SIR_algo(data_db, predict_range=predict_range, s_0=s_0, threshConfrirm=threshConfrirm)
        sir_annot = dict(xref='paper', yref='paper', x=0.25, y=0.95, align='left', font=dict(size=14), text=text)
        data['Date'] = data.index
        data.Date = pd.to_datetime(data.Date)
        with open(os.path.join(os.getcwd(), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
            fsc1 = scatter_country_plot(data, fname=' - ' + base_country + ' - Prediction with SIR Algorithm ',
                                        inputs=data.keys()[:-1], annotations=sir_annot, day=day.strftime('%d/%m/%y'))
            f.write(fsc1.to_html(full_html=False, include_plotlyjs='cdn'))
    except:
        print('Not executed Prediction with SIR Algorithm')

##################################################################################################################
# Prophet Algorithm
do_prophet = True
if do_prophet:
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
        # For Active
        act, forecast_act, fig_act = prophet_modeling_and_predicting(base_db, 'Active', predict_range=predict_range,
                                            first_n=predict_range, last_n=predict_range, threshConfrirm=threshConfrirm)

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
        prop_df['PredictedActive'] = dth.Deaths.values.clip(0).astype(int)
        prop_df['LowPredictedConfirmed'] = forecast_cnfrm.trend_lower.values.clip(0).astype(int)
        prop_df['LowPredictedRecovered'] = forecast_rec.trend_lower.values.clip(0).astype(int)
        prop_df['LowPredictedDeaths'] = forecast_dth.trend_lower.values.clip(0).astype(int)
        prop_df['LowPredictedActive'] = forecast_dth.trend_lower.values.clip(0).astype(int)
        prop_df['HighPredictedConfirmed'] = forecast_cnfrm.trend_upper.values.clip(0).astype(int)
        prop_df['HighPredictedRecovered'] = forecast_rec.trend_upper.values.clip(0).astype(int)
        prop_df['HighPredictedDeaths'] = forecast_dth.trend_upper.values.clip(0).astype(int)
        prop_df['HighPredictedActive'] = forecast_dth.trend_upper.values.clip(0).astype(int)
        prop_df = prop_df.fillna(0)

        prop_df = prop_df[prop_df['LowPredictedRecovered'] <= prop_df['LowPredictedConfirmed']]
        prop_df['Date'] = prop_df.index
        prop_df.Date = pd.to_datetime(prop_df.Date)

        Dprop = prop_df['Date'].max() - day

        # Future Ratio and percentages
        pr_pps = float(prop_df.LowPredictedRecovered[-1]/prop_df.LowPredictedConfirmed[-1])
        pd_pps = float(prop_df.LowPredictedDeaths[-1]/prop_df.LowPredictedConfirmed[-1])

        print("The percentage of Low Bound Predicted recovery after confirmation is " + str(round(pr_pps*100, 2)))
        print("The percentage of Low Bound Predicted Death after confirmation is " + str(round(pd_pps*100, 2)))
        print('Days to recovery ' + str(Dprop.days) + ' - ' + prop_df['Date'].max().strftime('%d/%m/%y'))

        pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.95, align='left', font=dict(size=14),
                        text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                             + '<br>Low Bound Predicted Recovery after confirmation is ' + str(round(pr_pps*100, 1))
                             + '%<br>Low Bound Predicted Death after confirmation is ' + str(round(pd_pps*100, 1)) + '%'
                             + '<br>Days to recovery ' + str(Dprop.days) + ' - ' + prop_df['Date'].max().strftime('%d/%m/%y'))

        with open(os.path.join(os.getcwd(), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
            fsc2 = line_country_plot(prop_df,
                                     fname=' - ' + base_country + ' - Prediction with Prophet Algorithm ',
                                     prefixes=['', 'Predicted', 'LowPredicted', 'HighPredicted'],
                                     annotations=pred_ann, day=day.strftime('%d/%m/%y'))
            if Dsir and Dprop.days > 2 * Dsir:
                try:
                    fsc2.show()
                except:
                    pass
            else:
                f.write(fsc2.to_html(full_html=False, include_plotlyjs='cdn'))
    except:
        print('Not executed Prediction with Prophet Algorithm')


###############################################################################################################
# Arima Algo - Autoregressive Integrated Moving Average Model
do_ARIMA = True
if do_ARIMA:
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

        aprop_df = aprop_df[aprop_df['PredictedRecovered'] <= aprop_df['PredictedConfirmed']]
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

        with open(os.path.join(os.getcwd(), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
            fsc3 = scatter_country_plot(aprop_df, fname=' - ' + base_country + ' - Prediction with ARIMA Algorithm ',
                                        inputs=aprop_df.keys()[:-1], annotations=pred_ann, day=day.strftime('%d/%m/%y'))
            if Dsir and Daprop.days > 2 * Dsir:
                try:
                    fsc3.show()
                except:
                    pass
            else:
                f.write(fsc3.to_html(full_html=False, include_plotlyjs='cdn'))
    except:
        print('Not executed Prediction with ARIMA Algorithm')

#########################################################################################
# LSTM
do_LSTM = True
if do_LSTM:
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
        dates = extend_index(data.Date, predict_range)
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
        cur_shape = lstm_prop_df.shape[0]
        lstm_prop_df = lstm_prop_df[lstm_prop_df['PredictedRecovered'] <= lstm_prop_df['PredictedConfirmed']]
        if lstm_prop_df.shape[0] >= cur_shape:
            lstm_prop_df = lstm_prop_df[lstm_prop_df['PredictedActive'] >= 0]
        # Future Ratio and percentages
        lpr_pps = float(lstm_prop_df.PredictedRecovered[-1]/lstm_prop_df.PredictedConfirmed[-1])
        lpd_pps = float(lstm_prop_df.PredictedDeaths[-1]/lstm_prop_df.PredictedConfirmed[-1])

        Dlstm = lstm_prop_df['Date'].max() - day

        print("The percentage of Predicted recovery after confirmation is %.2f" % (lpr_pps*100))
        print("The percentage of Predicted Death after confirmation is %.2f" % (lpd_pps*100))
        print('Days to recovery ' + str(Dlstm.days) + ' - ' + lstm_prop_df['Date'].max().strftime('%d/%m/%y'))

        pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.9, align='left', font=dict(size=14),
                        text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                             + '<br>Predicted Recovery after confirmation is ' + str(round(lpr_pps*100, 1))
                             + '%<br>Predicted Death after confirmation is ' + str(round(lpd_pps*100, 1))
                             + '<br>Days to recovery ' + str(Dlstm.days) + ' - ' + lstm_prop_df['Date'].max().strftime('%d/%m/%y'))

        with open(os.path.join(os.getcwd(), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
            fsc4 = scatter_country_plot(lstm_prop_df,
                                        fname=' - ' + base_country + ' - Prediction with LSTM Algorithm ',
                                        inputs=lstm_prop_df.keys()[:-1], annotations=pred_ann, day=day.strftime('%d/%m/%y'))
            if Dsir and Dlstm.days > 2 * Dsir:
                try:
                    fsc4.show()
                except:
                    pass
            else:
                f.write(fsc4.to_html(full_html=False, include_plotlyjs='cdn'))
    except:
        print('Not executed Prediction with LSTM Algorithm')

################################################################################################
# Regression
do_reg = True
if not do_on_pop:
    threshConfrirm = 1
if do_reg:
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
        cur_shape = reg_prop_df.shape[0]
        reg_prop_df = reg_prop_df[reg_prop_df['PredictedRecovered'] <= reg_prop_df['PredictedConfirmed']]
        if reg_prop_df.shape[0] >= cur_shape:
            reg_prop_df = reg_prop_df[reg_prop_df['PredictedActive'] >= 0]
        # Future Ratio and percentages
        rpr_pps = float(reg_prop_df.PredictedRecovered[-1]/reg_prop_df.PredictedConfirmed[-1])
        rpd_pps = float(reg_prop_df.PredictedDeaths[-1]/reg_prop_df.PredictedConfirmed[-1])

        Dreg = reg_prop_df['Date'].max() - day

        print("The percentage of Predicted recovery after confirmation is %.2f" % (rpr_pps*100))
        print("The percentage of Predicted Death after confirmation is %.2f" % (rpd_pps*100))
        print('Days to recovery ' + str(Dreg.days) + ' - ' + reg_prop_df['Date'].max().strftime('%d/%m/%y'))

        pred_ann = dict(xref='paper', yref='paper', x=0.2, y=0.9, align='left', font=dict(size=14),
                        text='Since the ' + str(threshConfrirm) + ' Confirmed Case'
                             + '<br>Predicted Recovery after confirmation is ' + str(round(rpr_pps*100, 1))
                             + '%<br>Predicted Death after confirmation is ' + str(round(rpd_pps*100, 1))
                             + '<br>Days to recovery ' + str(Dreg.days) + ' - ' + reg_prop_df['Date'].max().strftime('%d/%m/%y'))

        with open(os.path.join(os.getcwd(), day.strftime('%d%m%y') + '_' + base_country + '_Predictions .html'), 'a') as f:
            fsc5 = scatter_country_plot(reg_prop_df,
                                        fname=' - ' + base_country + ' - Prediction with Multi-layer Perceptron Regressor Algorithm ',
                                        inputs=reg_prop_df.keys()[:-1], annotations=pred_ann, day=day.strftime('%d/%m/%y'))
            if Dsir and Dreg.days > 2 * Dsir:
                try:
                    fsc5.show()
                except:
                    pass
            else:
                f.write(fsc5.to_html(full_html=False, include_plotlyjs='cdn'))
    except:
        print('Not executed Prediction with Multi-layer Perceptron Regressor Algorithm')
