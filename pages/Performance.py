from utils import *


def call_func():
    st.header("Performace of Various Models")
    
    #col1, col2 = st.columns(2)
    st.write('Time Series Model Comparison')
    df1 = pd.DataFrame([['Seasonal Naive Forecast model','23','78'],['Holt-Winters Model','16','105'],['ETS Model','11','65'],['SARIMA (one step ahead forecast)','1.3','8.27'],['SARIMA (dynamic forecast)','6','38']],columns=['Model Version','% MAPE','RMSE'])
    st.write(df1)

    st.write('Linear Regression Comparison')
    df = pd.DataFrame([['60-20-20','0.26','80'],['70-15-15','0.24','79'],['80-10-10','0.26','82'],['90-05-05','0.35','80']],columns=['Model Version','R Square','MAPE'])
    st.dataframe(df)
        
    
    


call_func()