from utils import *
# In[205]:

st.header("Holt Winter Model")

def call_model():
    
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
    st.write("Model Running...")
    hw_model = ExponentialSmoothing(train["close"],
                              trend    ="mul",
                              seasonal = "mul", 
                              seasonal_periods=365).fit()

    hw_fitted = hw_model.fittedvalues

    hw_resid = hw_model.resid

    #Adding the mean of the residuals to correct the bias.
    py_hw = hw_model.forecast(len(test["close"]))+np.mean(hw_resid)
    py_hw=pd.DataFrame(py_hw)
    py_hw.columns=['py_hw']
    py_hw=py_hw.reset_index()
    py_hw=py_hw.drop('index',axis=1)
    predictions1=test.copy()
    predictions1=predictions1.reset_index()
    py_hw2 = pd.concat([predictions1,py_hw],axis=1)
    py_hw2=py_hw2.set_index('date')
    
    st.write('Actual vs Predicted')
    st.write(py_hw2)
    
    
    st.write("Output Plot")
    train["close"].plot(figsize=(12,8), style="--", color="gray", legend=True, label="Train")
    hw_fitted.plot(color="b", legend=True, label="HW_Fitted")
    py_hw2["close"].plot(style="--",color="r", legend=True, label="Test")
    py_hw2["py_hw"].plot(color="b", legend=True, label="HW_Forecast")
    st.pyplot(plt)

    #MAE
    st.write('Mean Absolute Error :',mean_absolute_error(py_hw2.close, py_hw2["py_hw"]))
    st.write("")
    st.write('Root Mean Square Error :',math.sqrt(mean_squared_error(py_hw2.close, py_hw2["py_hw"])))


call_model()