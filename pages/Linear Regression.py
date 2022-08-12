from utils import *

st.header("Linear Regression")
st.write("Correlation Check")

data_merge4 = pd.read_csv('data/data_merge4.csv')
st.write("Data Set")
st.dataframe(data_merge4.head(2))

# Correlation matrix
data_merge6=data_merge4.copy()
matrix = data_merge6.corr()
st.write("Correlation Matrix")

#correlation matrix
corrmat = data_merge6.corr()
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
st.pyplot(plt)

#Setting X and y variables
x_features=['volume','text_neg','text_neu','text_pos','text_compound','twitter_text_neg','twitter_text_neu','twitter_text_pos','twitter_text_compound','close_14_ema_x','close_14_ema_y','Text_Perc_Contribution','TWT_Perc_Contribution']
scaler = StandardScaler()
non_imputed_num_transformer = Pipeline( steps = [('scaler', scaler)])
non_imputed_num_features=x_features

preprocessor = ColumnTransformer(transformers=[('num_not_imputed', non_imputed_num_transformer, non_imputed_num_features)])

# Lasso Regression for feature selection
x=data_merge6[x_features]
y=data_merge6[['close']]
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size = 0.7,random_state = 80)
lasso = Pipeline(steps=[('preprocessor', preprocessor),('regressor', Lasso(alpha = 0.01, max_iter = 500))])
lasso.fit( X_train, y_train )

column_names=non_imputed_num_features
st.write("Lasso Regression")
# lasso['regressor'].intercept_
# lasso['regressor'].coef_
features_df = pd.DataFrame(dict(zip(column_names,np.round(lasso['regressor'].coef_, 2))).items(),columns = ['feature', 'coefs'])
st.dataframe(features_df)
st.write("Column to be removed using Lasso Regression")
st.dataframe(features_df[features_df.coefs == 0])

y_pred_lasso = lasso.predict(X_test)
mse_v1 = mean_squared_error(y_test, y_pred_lasso)
st.write("Mean Square Error",mse_v1)
st.write("R Square ",r2_score(y_test, y_pred_lasso))
rmse_v1 = np.sqrt(mse_v1)
st.write("RMSE ",rmse_v1)

################################################################################

st.header("Linear Regression (60:20:20)")

x3_features=['volume','text_compound','twitter_text_compound','Text_Perc_Contribution','TWT_Perc_Contribution']
x=data_merge6[x3_features]
y=data_merge6[['close']]
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.4, random_state=60)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=60) 

# In[369]:


# Create linear regression object
lr0 = LinearRegression()
st.write("Running the Linear Regression")
# Train the model using the training set
lr0.fit(x_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = lr0.predict(x_train)
y_val_pred = lr0.predict(x_val)


# Print empirical risk on both sets
st.write("MSE on training set:",mean_squared_error(y_train, y_train_pred))
st.write('MSE on validation set:',mean_squared_error(y_val, y_val_pred))
st.write('')

# Print R squared on both sets
st.write('R squared on training set:',r2_score(y_train, y_train_pred))
st.write('R squared on validation set:',r2_score(y_val, y_val_pred))
st.write('')

# In[370]:



lr0.fit(x_test,y_test)
y_test_pred = lr0.predict(x_test)
st.write('MSE on test set:',mean_squared_error(y_test, y_test_pred))
st.write('RMSE on test set:',math.sqrt(mean_squared_error(y_test, y_test_pred)))
st.write('R squared on test set:',r2_score(y_test, y_test_pred))
st.write('')
# In[371]:


y_test2 = y_test['close'].tolist()
y_test_pred2 = y_test_pred.tolist()
y_test_pred3=[]
for i in range(0,len(y_test_pred2)):
    k=y_test_pred2[i]
    res = str(k)[1:-1]
    n=float(res)
    y_test_pred3.append(n)
    
y_df = pd.DataFrame({"actual": y_test2,
                     "predicted": y_test_pred3})
y_df['residual']=y_df['actual']-y_df['predicted']
st.write("Actual vs Predicted vs Residual")
st.dataframe(y_df)


# In[372]:
plt.clf()
st.write("Fit Chart")
# creating fit chart
y_df2=y_df.reset_index()
plt.figure(figsize=(6, 6))
plt.plot(y_df2['index'],y_df2['actual'])
plt.plot(y_df2['index'],y_df2['predicted'])
plt.plot(y_df2['index'],y_df2['residual'])
#plt.title('Fit Chart')
plt.xlabel("Index")
plt.ylabel("actual")
plt.gca().legend(('actual','predicted','residual'))
plt.tight_layout()
st.pyplot(plt)

# In[377]:

st.write("Coefficient from model")
s=lr0.coef_
col =s.tolist()
coef=col[0]
coef_dict={}
for i in range(0,len(x3_features)):
    dict2={x3_features[i]:coef[i]}
    coef_dict={**coef_dict,**dict2}
st.write(pd.DataFrame([coef_dict]))


################################################################################

st.header("Linear Regression (70:15:15)")

x3_features=['volume','text_compound','twitter_text_compound','Text_Perc_Contribution','TWT_Perc_Contribution']
x=data_merge6[x3_features]
y=data_merge6[['close']]
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.3, random_state=60)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=60) 

# In[369]:


# Create linear regression object
lr0 = LinearRegression()
st.write("Running the Linear Regression")
# Train the model using the training set
lr0.fit(x_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = lr0.predict(x_train)
y_val_pred = lr0.predict(x_val)


# Print empirical risk on both sets
st.write("MSE on training set:",mean_squared_error(y_train, y_train_pred))
st.write('MSE on validation set:',mean_squared_error(y_val, y_val_pred))
st.write('')

# Print R squared on both sets
st.write('R squared on training set:',r2_score(y_train, y_train_pred))
st.write('R squared on validation set:',r2_score(y_val, y_val_pred))
st.write('')

# In[370]:



lr0.fit(x_test,y_test)
y_test_pred = lr0.predict(x_test)
st.write('MSE on test set:',mean_squared_error(y_test, y_test_pred))
st.write('RMSE on test set:',math.sqrt(mean_squared_error(y_test, y_test_pred)))
st.write('R squared on test set:',r2_score(y_test, y_test_pred))
st.write('')
# In[371]:


y_test2 = y_test['close'].tolist()
y_test_pred2 = y_test_pred.tolist()
y_test_pred3=[]
for i in range(0,len(y_test_pred2)):
    k=y_test_pred2[i]
    res = str(k)[1:-1]
    n=float(res)
    y_test_pred3.append(n)
    
y_df = pd.DataFrame({"actual": y_test2,
                     "predicted": y_test_pred3})
y_df['residual']=y_df['actual']-y_df['predicted']
st.write("Actual vs Predicted vs Residual")
st.dataframe(y_df)


# In[372]:
plt.clf()
st.write("Fit Chart")
# creating fit chart
y_df2=y_df.reset_index()
plt.figure(figsize=(6, 6))
plt.plot(y_df2['index'],y_df2['actual'])
plt.plot(y_df2['index'],y_df2['predicted'])
plt.plot(y_df2['index'],y_df2['residual'])
#plt.title('Fit Chart')
plt.xlabel("Index")
plt.ylabel("actual")
plt.gca().legend(('actual','predicted','residual'))
plt.tight_layout()
st.pyplot(plt)

# In[377]:

st.write("Coefficient from model")
s=lr0.coef_
col =s.tolist()
coef=col[0]
coef_dict={}
for i in range(0,len(x3_features)):
    dict2={x3_features[i]:coef[i]}
    coef_dict={**coef_dict,**dict2}
st.write(pd.DataFrame([coef_dict]))



################################################################################

st.header("Linear Regression (80:10:10)")

x3_features=['volume','text_compound','twitter_text_compound','Text_Perc_Contribution','TWT_Perc_Contribution']
x=data_merge6[x3_features]
y=data_merge6[['close']]
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=60)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=60) 

# In[369]:


# Create linear regression object
lr0 = LinearRegression()
st.write("Running the Linear Regression")
# Train the model using the training set
lr0.fit(x_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = lr0.predict(x_train)
y_val_pred = lr0.predict(x_val)


# Print empirical risk on both sets
st.write("MSE on training set:",mean_squared_error(y_train, y_train_pred))
st.write('MSE on validation set:',mean_squared_error(y_val, y_val_pred))
st.write('')

# Print R squared on both sets
st.write('R squared on training set:',r2_score(y_train, y_train_pred))
st.write('R squared on validation set:',r2_score(y_val, y_val_pred))
st.write('')

# In[370]:



lr0.fit(x_test,y_test)
y_test_pred = lr0.predict(x_test)
st.write('MSE on test set:',mean_squared_error(y_test, y_test_pred))
st.write('RMSE on test set:',math.sqrt(mean_squared_error(y_test, y_test_pred)))
st.write('R squared on test set:',r2_score(y_test, y_test_pred))
st.write('')
# In[371]:


y_test2 = y_test['close'].tolist()
y_test_pred2 = y_test_pred.tolist()
y_test_pred3=[]
for i in range(0,len(y_test_pred2)):
    k=y_test_pred2[i]
    res = str(k)[1:-1]
    n=float(res)
    y_test_pred3.append(n)
    
y_df = pd.DataFrame({"actual": y_test2,
                     "predicted": y_test_pred3})
y_df['residual']=y_df['actual']-y_df['predicted']
st.write("Actual vs Predicted vs Residual")
st.dataframe(y_df)

# In[372]:
plt.clf()
st.write("Fit Chart")
# creating fit chart
y_df2=y_df.reset_index()
plt.figure(figsize=(6, 6))
plt.plot(y_df2['index'],y_df2['actual'])
plt.plot(y_df2['index'],y_df2['predicted'])
plt.plot(y_df2['index'],y_df2['residual'])
#plt.title('Fit Chart')
plt.xlabel("Index")
plt.ylabel("actual")
plt.gca().legend(('actual','predicted','residual'))
plt.tight_layout()
st.pyplot(plt)

# In[377]:

st.write("Coefficient from model")
s=lr0.coef_
col =s.tolist()
coef=col[0]
coef_dict={}
for i in range(0,len(x3_features)):
    dict2={x3_features[i]:coef[i]}
    coef_dict={**coef_dict,**dict2}
st.write(pd.DataFrame([coef_dict]))