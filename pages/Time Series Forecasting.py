from utils import *


def sarima_get_stock_prices_ohlc(sym):
    
    sbin = get_history(symbol=sym,
                       start=date(2022,8,2),
                       end=date(2022,8,12))
    sbin = sbin.reset_index()
    sbin = sbin[['Date','Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume']]
    if sbin.shape[0]:
        sbin['Date'] = pd.to_datetime(sbin['Date'],format='%Y-%m-%d')
        sbin.sort_values(by='Date', ascending=False)
    return sbin

def sarima_grid_search(y,seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
    
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

#                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    return param_mini, param_seasonal_mini, mini

# Call this function after pick the right(p,d,q) for SARIMA based on AIC               
def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
    # fit the model 
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    # st.write(results.summary().tables[1])
    st.write("Summary SARIMA")
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    st.pyplot(plt)
    
    
    
    # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
    # meaning that forecasts at each point are generated using the full history up to that point.
    plt.clf()
    pred = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    
    k=mean_absolute_error(y_forecasted, y_to_test)
    n=np.mean(np.abs((y_to_test - y_forecasted) / y_to_test)) * 100
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    st.write("Mean Absolute error :",k)
    st.write("MAPE is :",n)
    st.write('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))
    
    st.write("")
    st.write("Outcome with one step ahead forecast")
    ax = y.plot(label='observed')
    y_forecasted.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    # A better representation of our true predictive power can be obtained using dynamic forecasts. 
    # In this case, we only use information from the time series up to a certain point, 
    # and after that, forecasts are generated using values from previous forecasted time points.
    plt.clf()
    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    
    k_dynamic=mean_absolute_error(y_forecasted_dynamic, y_to_test)
    n_dynamic=np.mean(np.abs((y_to_test - y_forecasted_dynamic) / y_to_test)) * 100
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    st.write("Mean Absolute error :",k_dynamic)
    st.write("MAPE is :",n_dynamic)
    st.write('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))
    
    st.write("")
    st.write("Outcome with Dynamic forecast")
    ax = y.plot(label='observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')

    plt.legend()
    plt.show()
    st.pyplot(plt)
    
    return (results)


def forecast(model,predict_steps,y):
    
    pred_uc = model.get_forecast(steps=60)
    pred_ci = pred_uc.conf_int()
    pci = pred_ci.reset_index()
    pci.columns = ['ID','Lower Bound','Upper Bound']
    pci['dates'] = pd.Timestamp('2022-08-01')
    pci['dates2']=pci['dates']
    for i in range(0,len(pci)):
        pci['dates2'][i]=pci['dates'][i]+datetime.timedelta(days=i+1)

    pci=pci.drop(['ID','dates'],axis=1)
    pci=pci.rename(columns={'dates2':'Date'})

    plt.clf()
    st.write("Future Forecasting Outcome")
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['ID','Predicted_Mean']
    pm['dates'] = pd.Timestamp('2022-08-01')
    pm['dates2']=pm['dates']
    for i in range(0,len(pm)):
        pm['dates2'][i]=pm['dates'][i]+datetime.timedelta(days=i+1)

    pm=pm.drop(['ID','dates'],axis=1)
    pm=pm.rename(columns={'dates2':'Date'})

    pci2=pci.set_index('Date')
    pm2=pm.set_index('Date')
    
    ax = y.plot(label='observed', figsize=(14, 7))
    pm2.plot(ax=ax, label='Forecast')
    ax.fill_between(pci2.index,
                    pci2.iloc[:, 0],
                    pci2.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)

    plt.legend()
    plt.show()
    st.pyplot(plt)

    final_table = pm.join(pci.set_index('Date'), on='Date')
    final_table=final_table.set_index('Date')
    st.write("Predicted Outcome from Future Forecasting")
    st.write(final_table)
    return (final_table)


st.header("Time Series Prediction")

def call_forecast_model():
    start = time.time()
    data_merge6 = pd.read_csv('data/data_merge4.csv')
    #st.write("Data Set")
    #st.dataframe(data_merge6.head(2))
    st.write('Model Running...')

    df = data_merge6[['date','close']]
    df.date = pd.to_datetime(df.date)
    df = df.set_index("date")
    df['close'].plot(style="-")


    # In[595]:


    y=df['close']
    y_to_train = y[:'2021-06-22'] # dataset to train
    y_to_val = y['2021-06-23':] # last X months for test  
    predict_date = len(y) - len(y[:'2021-06-23'])

    # param_mini, param_seasonal_mini, mini = sarima_grid_search(y,60)
    # st.write('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))

    model = sarima_eva(y,(0, 1, 1),(0, 1, 1, 60),60,'2021-06-23',y_to_val)

    forecast(model,60,y)
    
    
    st.write("Comparison with current actual price")
    # list_of_stocks = ['RELIANCE']
    # df_data = pd.DataFrame()
    # for i in list_of_stocks:
    #     df1 = sarima_get_stock_prices_ohlc(i)
    #     df_data = df_data.append(df1)
    # df_data=df_data[['Date','Close']]
    # df_data.to_csv('data/time_series_data_reliance.csv')
    df_data = pd.read_csv('data/time_series_data_reliance.csv')
    
    pred_uc = model.get_forecast(steps=60)
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['ID','Predicted_Mean']
    pm['dates'] = pd.Timestamp('2022-08-01')
    pm['dates2']=pm['dates']
    for i in range(0,len(pm)):
        pm['dates2'][i]=pm['dates'][i]+datetime.timedelta(days=i+1)

    pm=pm.drop(['ID','dates'],axis=1)
    pm=pm.rename(columns={'dates2':'Date'})

    df_final=pd.merge(df_data,pm,left_on="Date",right_on="Date",how="left")
    st.write(df_final)
    
    st.write("Accuracy check")
    k=np.mean(np.abs((df_final.Close - df_final["Predicted_Mean"]) / df_final.Close)) * 100
    st.write("MAPE is",k)
    
    print(start - time.time())

call_forecast_model()

