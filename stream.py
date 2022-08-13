from utils import *

def main():
    st.header("Know your stock future")
    
    col1, col2 = st.columns(2)
    
    with col1:
        option = st.selectbox('Select a Stock',('RELIANCE',''))
        #st.write('You selected:', option)
    
    with col2:
        option1 = st.selectbox('Select Model',('Linear Regression',''))
        #st.write('You selected:', option1)
    
    if st.button('Predict'):
        st.write('Please Wait...')
        out = get_predictions(option,option1)
        
if __name__ == '__main__':
    main()
