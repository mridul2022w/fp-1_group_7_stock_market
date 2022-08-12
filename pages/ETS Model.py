from utils import *

st.header("ETS Model")

def call_ets_model():
    
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
    
    ets_AAdA=sm.tsa.statespace.ExponentialSmoothing(train["close"],
                                               trend=True, 
                                               initialization_method= 'concentrated', 
                                               seasonal=60, 
                                               damped_trend=True).fit()

    fc_AAdA=ets_AAdA.forecast(len(test))

    fc_AAdA=pd.DataFrame(fc_AAdA)
    fc_AAdA.columns=['fc_AAdA']
    fc_AAdA=fc_AAdA.reset_index()
    fc_AAdA=fc_AAdA.drop('index',axis=1)
    predictions2=test.copy()
    predictions2=predictions2.reset_index()
    fc_AAdA2 = pd.concat([predictions2,fc_AAdA],axis=1)
    fc_AAdA2=fc_AAdA2.set_index('date')
    
    st.write('Actual vs Predicted')
    st.write(fc_AAdA2)


    # In[214]:


    #Plotting
    st.write("Output Plot")
    train["close"].plot(figsize=(12,8), style="--", color="gray", legend=True, label="Train")
    ets_AAdA.fittedvalues.plot(color="b", legend=True, label="AAdA_Fitted")
    fc_AAdA2["close"].plot(style="--",color="r", legend=True, label="Test")
    fc_AAdA2["fc_AAdA"].plot(color="b", legend=True, label="AAdA_Forecast")
    st.pyplot(plt)
    
    
    st.write('Mean Absolute Error :',mean_absolute_error(fc_AAdA2.close, fc_AAdA2["fc_AAdA"]))
    
    st.write('Root Mean Square Error :',math.sqrt(mean_squared_error(fc_AAdA2.close, fc_AAdA2["fc_AAdA"])))
    
    
call_ets_model()