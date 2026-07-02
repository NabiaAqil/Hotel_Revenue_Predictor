import streamlit as st
import pickle
import pandas as pd

try:
    with open('hotel_revenue_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    st.error("Model file 'hotel_revenue_model.pkl' nahi mili!")

st.title('🏨 Pakistan Hotel Revenue Prediction')
st.divider()

col1, col2 = st.columns(2)

with col1:
    month = st.slider('Select Month (1-12):', min_value=1, max_value=12, value=6)
    avg_occupancy = st.number_input('Average Occupancy %:', min_value=0.0, max_value=100.0, value=60.0)
    room_rate = st.number_input('Room Rate in PKR:', min_value=1000.0, value=15000.0)
    total_rooms = st.number_input('Total Rooms:', min_value=1, value=50)

with col2:
    staff_count = st.number_input('Staff Count:', min_value=1, value=20)
    amenities_count = st.slider('Amenities Count:', min_value=0, max_value=20, value=5)
    customer_rating = st.slider('Customer Rating (1-5):', min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    avg_stay = st.number_input('Average Stay Duration (Days):', min_value=1.0, value=2.0)

province = st.selectbox('Select Province:', ['Punjab', 'Sindh', 'KPK', 'Balochistan'])
city = st.selectbox('Select City:', ['Karachi', 'Lahore', 'Islamabad', 'Multan', 'Peshawar', 'Quetta'])
booking_source = st.selectbox('Booking Source:', ['Booking.com', 'Travel Agency', 'Walk-in', 'Website'])
season = st.selectbox('Select Season:', ['Winter', 'Spring', 'Summer', 'Autumn'])

if st.button("Predict Hotel Revenue 🔮", use_container_width=True):
    
    raw_data = pd.DataFrame([{
        'Month': month,
        'Avg_Occupancy_%': avg_occupancy,
        'Room_Rate_PKR': room_rate,
        'Total_Rooms': total_rooms,
        'Staff_Count': staff_count,
        'Amenities_Count': amenities_count,
        'Customer_Rating': customer_rating,
        'Avg_Stay_Duration_Days': avg_stay,
        'Province': province,
        'City': city,
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

        prediction = model.predict(final_data)
        st.success(f" Estimated Hotel Revenue: PKR {prediction[0]:,.2f}")
        
    except AttributeError:
        st.error("Model ke andar feature names save nahi hain.")
    except Exception as e:
        st.error(f"Error: {e}")
