from utils import *


st.header("Seasonal Naive Forecast Model")

def pysnaive(train_series,seasonal_periods,forecast_horizon):
    '''
    Python implementation of Seasonal Naive Forecast. 
    This should work similar to https://otexts.com/fpp2/simple-methods.html
    Returns two arrays
     > fitted: Values fitted to the training dataset
     > fcast: seasonal naive forecast
    
    Author: Sandeep Pawar
    
    Date: Apr 9, 2020
    
    Ver: 1.0
    
    train_series: Pandas Series
        Training Series to be used for forecasting. This should be a valid Pandas Series. 
        Length of the Training set should be greater than or equal to number of seasonal periods
        
    Seasonal_periods: int
        No of seasonal periods
        Yearly=1
        Quarterly=4
        Monthly=12
        Weekly=52
        

    Forecast_horizon: int
        Number of values to forecast into the future
    
    e.g. 
    fitted_values = pysnaive(train,12,12)[0]
    fcast_values = pysnaive(train,12,12)[1]
    '''
    
    if len(train_series)>= seasonal_periods: #checking if there are enough observations in the training data
        
        last_season=train_series.iloc[-seasonal_periods:]
        
        reps=np.int(np.ceil(forecast_horizon/seasonal_periods))
        
        fcarray=np.tile(last_season,reps)
        
        fcast=pd.Series(fcarray[:forecast_horizon])
        
        fitted = train_series.shift(seasonal_periods)
        
    else:
        fcast=print("Length of the trainining set must be greater than number of seasonal periods") 
    
    return fitted, fcast


# In[83]:

def call_seasonal_Navie_Forecast_Model():
    
    data_merge6 = pd.read_csv('data/data_merge4.csv')
    
    df = data_merge6[['date','close']]
    df.date = pd.to_datetime(df.date)
    df = df.set_index("date")
    df['close'].plot(style="-")

    y=df['close']
    y_to_train = y[:'2021-06-23'] # dataset to train
    y_to_val = y['2021-06-23':] # last X months for test  
    predict_date = len(y) - len(y[:'2021-06-23']) 
    test=pd.DataFrame(y_to_val)
    train=pd.DataFrame(y_to_train)
    
    #Fitted values
    py_snaive_fit = pysnaive(train["close"], 
                         seasonal_periods=365,
                         forecast_horizon=len(train))[0]

    #forecast
    py_snaive = pysnaive(train["close"], 
                         seasonal_periods=365,
                         forecast_horizon=len(train))[1]

    #Residuals
    py_snaive_resid = (train["close"] - py_snaive_fit).dropna()



    predictions=train.copy()
    predictions["py_snaive"] = py_snaive.values
    
    st.write("Predictions")
    st.write(predictions)


    # In[84]:


    pd.plotting.register_matplotlib_converters()
    train["close"].plot(figsize=(12,8))#, style="--", color="gray", legend=True, label="Train")
    py_snaive_fit.plot(color="b", legend=True, label="SNaive_Fitted")
    predictions["close"].plot(style="--",color="r", legend=True, label="Test")
    #predictions["py_snaive"].plot(color="b", legend=True, label="Snaive_fc");
    st.pyplot(plt)


    # In[85]:


    #Training score (MAE)
    mean_absolute_error(train["close"].iloc[-len(py_snaive_fit.dropna()):], py_snaive_fit.dropna())
    st.write("Mean Square Error (train data):",mean_absolute_error(train["close"].iloc[-len(py_snaive_fit.dropna()):], py_snaive_fit.dropna()))


    # In[86]:


    #RSME
    math.sqrt(mean_squared_error(train["close"].iloc[-len(py_snaive_fit.dropna()):], py_snaive_fit.dropna()))
    st.write("Root Mean Square Error (train data):",math.sqrt(mean_squared_error(train["close"].iloc[-len(py_snaive_fit.dropna()):], py_snaive_fit.dropna())))

    # In[87]:


    #Test score (MAE)
    mean_absolute_error(predictions["close"], predictions["py_snaive"])
    st.write("Mean Square Error (test data):",mean_absolute_error(predictions["close"], predictions["py_snaive"]))


    # In[88]:


    #RSME
    st.write("Root Mean Square Error (test data):",math.sqrt(mean_squared_error(predictions["close"], predictions["py_snaive"])))


call_seasonal_Navie_Forecast_Model()