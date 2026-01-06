import pandas as pd
import joblib
import os
import sys

def predict_house_price():
    print("---------------------------------------------------------")
    print("       House Price Prediction - Interactive Tool         ")
    print("---------------------------------------------------------")
    
    # 1. Load Model and Encoder
    model_path = os.path.join('model_training', 'PKL_files/rf_model.pkl')
    encoder_path = os.path.join('model_training', 'PKL_files/label_encoder.pkl')
    
    # Handle running from subfolder
    if not os.path.exists(model_path):
        model_path = 'PKL_files/rf_model.pkl'
        encoder_path = 'PKL_files/label_encoder.pkl'
        
    print("Loading saved model...")
    try:
        rf_model = joblib.load(model_path)
        le_ocean = joblib.load(encoder_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Saved model not found. Please run the Jupyter Notebook first to train and save the model.")
        return

    print("---------------------------------------------------------")
    print("Please enter the details of the house to predict its price.")

    try:
        # 2. Get User Input
        # Map friendly names to actual dataset values
        location_options = {
            "1": ("<1H OCEAN", "🚗 <1H OCEAN"),
            "2": ("INLAND", "🌾 INLAND"),
            "3": ("ISLAND", "🏝️ ISLAND"),
            "4": ("NEAR BAY", "🌉 NEAR BAY"),
            "5": ("NEAR OCEAN", "🌊 NEAR OCEAN")
        }

        print("\nSelect the Location Area:")
        for key, value in location_options.items():
            print(f"{key}. {value[1]}")

        choice = input("Enter your choice (1-5): ").strip()
        while choice not in location_options:
            print("Invalid choice. Please enter a number between 1 and 5.")
            choice = input("Enter your choice (1-5): ").strip()
            
        ocean_input = location_options[choice][0]
        print(f"Selected: {location_options[choice][1]}")

        ocean_encoded = le_ocean.transform([ocean_input])[0]
        
        # Helper to get float input
        def get_float(prompt, default=None):
            val = input(f"{prompt}: ")
            if not val and default is not None:
                return default
            return float(val)

        print("\n--- Property Details ---")
        total_rooms = get_float("Total Rooms (e.g., 2000)")
        total_bedrooms = get_float("Total Bedrooms (e.g., 400)")
        households = get_float("Households (e.g., 350)")
        median_income = get_float("Median Income (1 = $10k, e.g., 3.5 for $35k)")
        housing_median_age = get_float("Median Age of House (e.g., 25)")
        
        print("\n--- Location Details (Optional - Press Enter for defaults) ---")
        # Hardcoded defaults or safe fallbacks since we don't have the full dataframe loaded here
        # These are averages from the EDA
        defaults = {
            "<1H OCEAN": (-118.8, 34.5, 1520),
            "INLAND": (-119.7, 36.7, 1390),
            "ISLAND": (-118.3, 33.3, 669),
            "NEAR BAY": (-122.2, 37.8, 1230),
            "NEAR OCEAN": (-119.3, 34.7, 1354)
        }
        
        def_long, def_lat, def_pop = defaults.get(ocean_input, (-119.5, 35.6, 1425))
        
        longitude = get_float(f"Longitude [Default: {def_long:.2f}]", default=def_long)
        latitude = get_float(f"Latitude [Default: {def_lat:.2f}]", default=def_lat)
        population = get_float(f"Population in Block [Default: {def_pop:.0f}]", default=def_pop)

        # 3. Prepare Input
        # Columns: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity_encoded
        columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity_encoded']
        
        input_data = pd.DataFrame([{
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity_encoded': ocean_encoded
        }])
        
        # Reorder columns to match training
        input_data = input_data[columns]

        # 4. Predict
        predicted_price = rf_model.predict(input_data)[0]
        
        print("---------------------------------------------------------")
        print(f"PREDICTED HOUSE PRICE: ${predicted_price:,.2f}")
        print("---------------------------------------------------------")
        
    except ValueError as e:
        print(f"Error: Invalid input. Please enter numbers where required. ({e})")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict_house_price()
