from utils import *

#if __name__ == '__main__':
    # list_of_stocks = ['RELIANCE','MARUTI']
    # col_names = ['text','twitter_text']
    # # data fetching
    # # df = data_fetch_and_merge_from_different_sources(list_of_stocks)
    # #df = pd.read_csv('data/data.csv')
    # 
    # # data cleaning
    # # cleaned_data = clean_the_data(df,col_names)
    # #cleaned_data = pd.read_csv('data/cleaned_data.csv')
    # 
    # # # sentiment analysis
    # # df_with_sentiments = sentiment_analysis(cleaned_data,col_names)
    # df_with_sentiments = pd.read_csv('data/data_with_sentiment.csv')
    # ##print(df_with_sentiments)
    # data12 = pd.DataFrame()
    # for i in list_of_stocks:
    #     data_new  = df_with_sentiments[df_with_sentiments['Symbol'] == i]
    #     data12 = data12.append(calculate_stock_indicators(data_new,'close_14_ema','close_14_ema'))
    # #print(data12)
    # df_final = ltm_on_the_text_data(data12,col_names)
    # print(df_final)
