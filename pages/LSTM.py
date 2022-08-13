from utils import *

st.header("LSTM Model")

def call_lstm():
    list_of_stocks = ['SBIN']
    df_final = pd.DataFrame()
    for i in list_of_stocks:
        df1 = get_stock_prices_ohlc(i)
        df_final = df_final.append(df1)
    df_final2=df_final.set_index('Date')


    # Plotting prices
    st.write("Price Visualization")
    plt.figure(figsize = (12,6))
    plt.plot(df_final2["Open"])
    plt.plot(df_final2["High"])
    plt.plot(df_final2["Low"])
    plt.plot(df_final2["Close"])
    plt.title('SBI stock price history')
    plt.ylabel('Price (USD)')
    plt.xlabel('Days')
    plt.legend(['Open','High','Low','Close'], loc='upper left')
    plt.show()
    st.pyplot(plt)


    # In[16]:

    plt.clf()
    # plotting volume
    st.write("Volume Visualization")
    plt.figure(figsize = (12,6))
    plt.plot(df_final2["Volume"])
    plt.title('SBI stock volume history')
    plt.ylabel('Volume')
    plt.xlabel('Days')
    plt.show()
    st.pyplot(plt)

    # In[19]:


    # Create a dataframe with only the Close Stock Price Column
    data_target = df_final.filter(['Close'])
    # Convert the dataframe to a numpy array to train the LSTM model
    target = data_target.values
    # Splitting the dataset into training and test
    # Target Variable: Close stock price value
    training_data_len = math.ceil(len(target)* 0.7) # training set has 75% of the data
    # Normalizing data before model fitting using MinMaxScaler
    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0,1))
    training_scaled_data = sc.fit_transform(target)

    # Create a training dataset containing the last 180-day closing price values we want to use to estimate the 181st closing price value.
    train_data = training_scaled_data[0:training_data_len  , : ]

    X_train = []
    y_train = []
    for i in range(180, len(train_data)):
        X_train.append(train_data[i-180:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train) # converting into numpy sequences to train the LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    st.write("Number of rows and columns for train data: ",X_train.shape)
    
    # We add the LSTM layer and later add a few Dropout layers to prevent overfitting.
    # Building a LTSM model with 50 neurons and 4 hidden layers. We add the LSTM layer with the following arguments:
    # 50 units which is the dimensionality of the output space
    # return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence input_shape as the shape of our training set.
    # When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped.
    # Thereafter, we add the Dense layer that specifies the output of 1 unit.
    # After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error.
    model = Sequential()
    st.write("Adding the first LSTM layer and some Dropout regularisation")
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    st.write("Adding a second LSTM layer and some Dropout regularisation")
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    st.write("Adding a third LSTM layer and some Dropout regularisation")
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    st.write("Adding a fourth LSTM layer and some Dropout regularisation")
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    st.write("Compiling the RNN")
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    st.write("Fitting the RNN to the Training set")
    # Fitting the RNN to the Training set
    
    #epoch to be set to 100
    model.fit(X_train, y_train, epochs = 30, batch_size = 32)

    # Getting the predicted stock price
    test_data = training_scaled_data[training_data_len - 180: , : ]

    #Create the x_test and y_test data sets
    X_test = []
    y_test =  target[training_data_len : , : ]
    for i in range(180,len(test_data)):
        X_test.append(test_data[i-180:i,0])

    # Convert x_test to a numpy array
    X_test = np.array(X_test)

    #Reshape the data into the shape accepted by the LSTM
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    st.write("Number of rows and columns for test data: ",X_test.shape)


    # In[24]:


    # Making predictions using the test dataset
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)


    # In[42]:
    train = data_target[:training_data_len]
    valid = data_target[training_data_len:]


    #charting outputs
    st.write("Actual vs Predicted")
    plt.clf()
    date=df_final[['Date']]
    date2 = date[:training_data_len]
    train2 = pd.concat([train, date2], axis=1)
    train2=train2.set_index('Date')
    date3 = date[training_data_len:]
    valid2 = pd.concat([valid, date3], axis=1)
    valid2=valid2.set_index('Date')
    valid2['Predictions'] = predicted_stock_price
    
    plt.figure(figsize=(10,5))
    plt.title('Model')
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Close Price USD ($)', fontsize=12)
    plt.plot(train2['Close'])
    plt.plot(valid2[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    st.pyplot(plt)


    # In[43]:


    #output data
    st.write(valid2.head(10))
    
    st.write("Accuracy Table")
    #MAE
    n=mean_absolute_error(valid2.Close, valid2["Predictions"])
    st.write("Mean Absolute error is",n)

    #MAPE
    k=np.mean(np.abs((valid2.Close - valid2["Predictions"]) / valid2.Close)) * 100
    
    
    #RMSE
    r=math.sqrt(mean_squared_error(valid2.Close, valid2["Predictions"]))
    st.write("RMSE is",r)

call_lstm()