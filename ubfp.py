import streamlit as st
import pickle
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium

# Load the StandardScaler
with open("D:/uber/pythonProject5/venv_name/Scripts/scaler.pkl", 'rb') as file:
    ss = pickle.load(file)

# Load the model from the pickle file
with open('D:/uber/pythonProject5/venv_name/Scripts/data.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar Navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to', ['Home', 'Predict'])

if selection == 'Home':
    st.markdown(
        """
        <div style='text-align: left;'>
            <h1 style='color: blue; font-size: 50px; margin-left: 0px;'>
                Uber Fare Prediction
            </h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='text-align: left; padding-left: 5px; font-size: 20px; color: red;'>
            <p>
                Welcome to the Uber Fare Prediction application! This tool allows you to estimate the fare of an Uber ride based on various factors 
                such as pickup location, drop-off location, and other ride attributes. Simply enter the details of your trip, and the app will predict 
                the fare amount using a machine learning model.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Add an image
    image_path = "C:/Users/Saambavi/OneDrive/Pictures/Saved Pictures/uber.jpg"  
    st.image(image_path, caption="Uber Fare Prediction Example", use_column_width=True)

if selection == 'Predict':
    # Function to get coordinates
    def get_coordinates(location):
        geolocator = Nominatim(user_agent="streamlit_app")
        coords = geolocator.geocode(location)
        return (coords.latitude, coords.longitude)

    st.title('Uber Fare Prediction')

    # Input fields
    pickup_address = st.text_input("Enter the starting location:", "New York, NY")
    dropoff_address = st.text_input("Enter the ending location:", "Washington, DC")
    passenger_count = st.number_input('Enter Passenger Count', min_value=1, max_value=10, value=1)
    day_of_week = st.selectbox('Select Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Determine car type based on passenger count
    if passenger_count <= 2:
        car_type = 'Mini'
    elif 2 < passenger_count <= 5:
        car_type = 'XUV'
    else:
        car_type = 'None'

    st.write(f"Selected Car Type: {car_type}")

    if pickup_address and dropoff_address:
        try:
            start_latlng = get_coordinates(pickup_address)
            end_latlng = get_coordinates(dropoff_address)

            # Calculate midpoint
            midpoint = [(start_latlng[0] + end_latlng[0]) / 2, (start_latlng[1] + end_latlng[1]) / 2]

            # Create map
            map_route = folium.Map(location=midpoint, zoom_start=6)

            # Add markers
            folium.Marker(start_latlng, tooltip="Start", popup=pickup_address).add_to(map_route)
            folium.Marker(end_latlng, tooltip="End", popup=dropoff_address).add_to(map_route)

            # Add route
            route = folium.PolyLine([start_latlng, end_latlng], color="blue", weight=2.5, opacity=1).add_to(map_route)

            # Display map in Streamlit
            st_folium(map_route, width=700, height=500)

            if st.button("Calculate Fare"):
                # Calculate the distance between pickup and dropoff locations
                distance_km = geodesic(start_latlng, end_latlng).kilometers

                # Prepare input features (ensure correct feature size)
                input_features = np.zeros((1, 24))

                # Fill in the known values
                input_features[0, 0] = start_latlng[1]  # pickup_longitude
                input_features[0, 1] = start_latlng[0]  # pickup_latitude
                input_features[0, 2] = end_latlng[1]  # dropoff_longitude
                input_features[0, 3] = end_latlng[0]  # dropoff_latitude
                input_features[0, 4] = passenger_count
                input_features[0, 12] = distance_km  # Ensure distance is placed correctly

                # One-hot encoding for car type
                if car_type == 'Mini':
                    input_features[0, 13] = 1
                elif car_type == 'XUV':
                    input_features[0, 14] = 1
                else:
                    input_features[0, 15] = 1

                # One-hot encoding for day of the week
                day_mapping = {'Monday': 16, 'Tuesday': 17, 'Wednesday': 18, 'Thursday': 19, 'Friday': 20, 'Saturday': 21, 'Sunday': 22}
                input_features[0, day_mapping[day_of_week]] = 1

                # Feature scaling
                input_features_scaled = ss.transform(input_features)

                # Predict fare using the model
                fare = model.predict(input_features_scaled)

                # Adjust fare for weekends
                if day_of_week in ['Saturday', 'Sunday']:
                    fare *= 1.2  # Increase by 20% for weekends

                # Adjust fare based on car type
                if car_type == 'XUV':
                    fare *= 1.5  # Increase by 50% for XUV
                elif car_type == 'None':
                    fare *= 2.0  # Double the fare for None car type

                st.write(f"Estimated Fare: ${fare[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
