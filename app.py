import streamlit as st
import pickle
import pandas as pd

# 1. Hotel Model file load karna (Yahan apne model file ka sahi naam check kar lein)
try:
    with open('hotel_revenue_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    st.error("Model file 'hotel_revenue_model.pkl' nahi mili! Pehle isay Colab mein upload/save karein.")

st.title('🏨 Pakistan Hotel Revenue Prediction')
st.divider()

# 2. User Inputs (Numeric)
col1, col2 = st.columns(2)

with col1:
    month = st.slider('Select Month (1-12):', min_value=1, max_value=12, value=6)
    avg_occupancy = st.number_input('Average Occupancy %:', min_value=0.0, max_value=100.0, value=60.0)
    room_rate = st.number_input('Room Rate in PKR:', min_value=1000.0, max_value=500000.0, value=15000.0)
    total_rooms = st.number_input('Total Rooms:', min_value=1, max_value=1000, value=50)

with col2:
    staff_count = st.number_input('Staff Count:', min_value=1, max_value=500, value=20)
    amenities_count = st.slider('Amenities Count:', min_value=0, max_value=20, value=5)
    customer_rating = st.slider('Customer Rating (1-5):', min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    avg_stay = st.number_input('Average Stay Duration (Days):', min_value=1.0, max_value=30.0, value=2.0)

# 3. User Inputs (Categorical Dropdowns)
province = st.selectbox('Select Province:', ['Punjab', 'Sindh', 'KPK', 'Balochistan'])
season = st.selectbox('Select Season:', ['Winter', 'Spring', 'Summer', 'Autumn'])

# 4. Inputs ko numbers (0 aur 1) mein convert karna (Jaise Miss ne sikhaya)
# Province Encoding
is_Punjab = 1 if province == "Punjab" else 0
is_Sindh = 1 if province == "Sindh" else 0
is_KPK = 1 if province == "KPK" else 0
is_Balochistan = 1 if province == "Balochistan" else 0

# Season Encoding
is_Winter = 1 if season == "Winter" else 0
is_Spring = 1 if season == "Spring" else 0
is_Summer = 1 if season == "Summer" else 0
is_Autumn = 1 if season == "Autumn" else 0

# 5. Predict Button
if st.button("Predict Hotel Revenue 🔮", use_container_width=True):
    # DataFrame banana bilkul training features ke columns ke mutabiq
    user_data = pd.DataFrame([{
        'Month': month,
        'Avg_Occupancy_%': avg_occupancy,
        'Room_Rate_PKR': room_rate,
        'Total_Rooms': total_rooms,
        'Staff_Count': staff_count,
        'Amenities_Count': amenities_count,
        'Customer_Rating': customer_rating,
        'Avg_Stay_Duration_Days': avg_stay,
        
        # Encoding wale columns
        'Province_Punjab': is_Punjab,
        'Province_Sindh': is_Sindh,
        'Province_KPK': is_KPK,
        'Province_Balochistan': is_Balochistan,
        'Season_Winter': is_Winter,
        'Season_Spring': is_Spring,
        'Season_Summer': is_Summer,
        'Season_Autumn': is_Autumn
    }])

    try:
        prediction = model.predict(user_data)
        st.success(f"### 💰 Estimated Hotel Revenue: PKR {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Columns Match Nahi Hue! Error: {e}")
