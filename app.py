import streamlit as st
import pickle
import pandas as pd

try:
    with open('hotel_revenue_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    st.error("File Not Found!")

st.title('🏨 Pakistan Hotel Revenue Prediction')
st.divider()

col1, col2 = st.columns(2)

with col1:
    month = st.slider('Select Month (1-12):', min_value=1, max_value=12, value=6)
    avg_occupancy = st.number_input('Average Occupancy %:', min_value=0.0, max_value=100.0, value=70.0)
    room_rate = st.number_input('Room Rate in PKR:', min_value=1000.0, value=15000.0)
    total_rooms = st.number_input('Total Rooms:', min_value=1, value=100)
    staff_count = st.number_input('Staff Count:', min_value=1, value=40)

with col2:
    amenities_count = st.slider('Amenities Count:', min_value=0, max_value=20, value=8)
    customer_rating = st.slider('Customer Rating (1-5):', min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    foreign_visitors = st.number_input('Number of Foreign Visitors:', min_value=0, value=50)
    local_visitors = st.number_input('Number of Local Visitors:', min_value=0, value=150)
    avg_stay = st.number_input('Average Stay Duration (Days):', min_value=1, value=3)

province = st.selectbox('Select Province:', ['Punjab', 'Sindh', 'KPK', 'Islamabad Capital Territory'])
city = st.selectbox('Select City:', ['Lahore', 'Karachi', 'Islamabad', 'Multan', 'Peshawar', 'Quetta', 'Rawalpindi'])
customer_type = st.selectbox('Customer Type:', ['Family', 'Individual', 'Tour Group'])
booking_source = st.selectbox('Booking Source:', ['Booking.com', 'Travel Agency', 'Walk-in', 'Website'])
season = st.selectbox('Select Season:', ['Winter', 'Spring', 'Summer', 'Autumn'])

if st.button("Predict Hotel Revenue ", use_container_width=True):

    raw_data = pd.DataFrame([{
        'Month': month,
        'Avg_Occupancy_%': avg_occupancy,
        'Room_Rate_PKR': room_rate,
        'Total_Rooms': total_rooms,
        'Staff_Count': staff_count,
        'Amenities_Count': amenities_count,
        'Customer_Rating': customer_rating,
        'Foreign_Visitors': foreign_visitors,
        'Local_Visitors': local_visitors,
        'Avg_Stay_Duration_Days': avg_stay,
        'Province': province,
        'City': city,
        'Customer_Type': customer_type,
        'Booking_Source': booking_source,
        'Season': season
    }])

    encoded_data = pd.get_dummies(raw_data, dtype=int)

    try:
        model_features = model.feature_names_in_

        for col in model_features:
            if col not in encoded_data.columns:
                encoded_data[col] = 0

        final_data = encoded_data[model_features]

        prediction = model.predict(final_data)[0]

        if prediction < 0:
            prediction = 0.0

        st.success(f" Estimated Hotel Revenue: PKR {prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
