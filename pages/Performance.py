from utils import *


def call_func():
    st.header("Performace of Various Models")
    
    #col1, col2 = st.columns(2)
    st.write('Time Series Model Comparison')
    df1 = pd.DataFrame([['Seasonal Naive Forecast model','70','908'],['Holt-Winters Model','20','629'],['ETS Model','10','305'],['SARIMA (one step ahead forecast)','1.3','43.13'],['SARIMA (dynamic forecast)','6','173']],columns=['Model Version','% MAPE','RMSE'])
    st.write(df1)

    st.write('Linear Regression Comparison')
    df = pd.DataFrame([['60-20-20','63','338'],['70-15-15','59','341'],['80-10-10','65','340'],['90-05-05','74','300'],['Best Fit','98','58']],columns=['Model Version','R Square','MAPE'])
    st.dataframe(df)
        
    
    


call_func()