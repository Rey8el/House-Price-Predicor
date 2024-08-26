import streamlit as st
import pandas as pd
import pickle

# Load the data and model
df = pd.read_csv('cleaned_data.csv')
locations = sorted(df['location'].unique())

# Load your pre-trained model (make sure to have your model file in the same directory)
with open('RidgeModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict price
def predict_price(location, bhk, bath, sqft):
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_data)[0]
    return prediction*100000

# Streamlit app
def main():
    st.title('Real Estate Price Prediction')

    # Location dropdown
    location = st.selectbox('Select Location', locations)
    
    # Input fields
    bhk = st.number_input('Enter number of BHK', min_value=1, max_value=10, value=2)
    bath = st.number_input('Enter number of Bathrooms', min_value=1, max_value=10, value=2)
    sqft = st.number_input('Enter total square feet', min_value=300, max_value=10000, value=1000)
    
    # Prediction
    if st.button('Predict Price'):
        price = predict_price(location, bhk, bath, sqft)
        st.success(f'The predicted price is ₹{price:,.2f}')
        
if __name__ == '__main__':
    main()
