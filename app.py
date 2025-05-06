import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import json
import os
import hashlib

# -------------------------
# User Data Persistence
# -------------------------
USER_FILE = "users.json"
HISTORY_DIR = "user_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users_dict):
    with open(USER_FILE, "w") as f:
        json.dump(users_dict, f, indent=4)

# -------------------------
# Session Initialization
# -------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'users' not in st.session_state:
    st.session_state['users'] = load_users()

# -------------------------
# Global Style
# -------------------------
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #e7e3df;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #5cdbbc;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            margin-top: 10px;
        }
        .stDownloadButton>button {
            background-color: #2196F3;
            color: white;
            border-radius: 8px;
            margin-top: 15px;
        }
        .stTabs [role="tab"] {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Login/Register Tabs
# -------------------------
def show_login_register():
    st.markdown("<div class='main-title'>🏡 House Price Predictor</div>", unsafe_allow_html=True)

    if st.session_state.get('show_registered_success'):
        st.toast("🎉 Registration completed successfully. Please log in!", icon="✅")
        st.session_state['show_registered_success'] = False

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        username = st.text_input("Username", key="login_username_input")
        password = st.text_input("Password", type="password", key="login_password_input")
        if st.button("Login", use_container_width=True):
            hashed_input_pass = hash_password(password)
            if st.session_state['users'].get(username) == hashed_input_pass:
                st.session_state['logged_in'] = True
                st.session_state['login_user'] = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("Choose a Username", key="reg_user")
        new_pass = st.text_input("Choose a Password", type="password", key="reg_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
        if st.button("Register", use_container_width=True):
            if new_user in st.session_state['users']:
                st.error("Username already exists.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            elif len(new_user.strip()) == 0 or len(new_pass.strip()) == 0:
                st.error("Username and password cannot be empty.")
            else:
                hashed_pass = hash_password(new_pass)
                st.session_state['users'][new_user] = hashed_pass
                save_users(st.session_state['users'])
                st.success("Registration successful! You can now log in.")
                st.toast(f"🎉 User '{new_user}' registered successfully!", icon="✅")
                st.balloons()
                st.session_state['show_registered_success'] = True
                st.rerun()

# -------------------------
# Load Model and Data
# -------------------------
def load_model_and_data():
    data = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['price'] = data.target
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model, df, X_train, y_train, X_test, y_test

# -------------------------
# Location Dataset Loading
# -------------------------
def clean_location_df(df):
    df = df.rename(columns=lambda x: x.strip())
    df.columns = df.columns.str.title()
    return df[['Country', 'State', 'City']]

df1 = clean_location_df(pd.read_csv("data/dataset1.csv"))
df2 = clean_location_df(pd.read_csv("data/dataset2.csv"))
df3 = clean_location_df(pd.read_csv("data/dataset3.csv"))

combined_locations = pd.concat([df1, df2, df3], ignore_index=True)
combined_locations.dropna(subset=['Country', 'State', 'City'], inplace=True)

countries = sorted(combined_locations['Country'].unique())

states_by_country = {
    country: sorted(combined_locations[combined_locations['Country'] == country]['State'].unique())
    for country in countries
}

cities_by_state = {
    (row['Country'], row['State']): sorted(
        combined_locations[
            (combined_locations['Country'] == row['Country']) &
            (combined_locations['State'] == row['State'])
        ]['City'].unique()
    )
    for _, row in combined_locations[['Country', 'State']].drop_duplicates().iterrows()
}

# -------------------------
# Pages
# -------------------------
def show_prediction_page(model):
    st.subheader("📊 Predict a New House Price")
    with st.expander("Enter House Features", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            MedInc_dollars = st.number_input("Median Income ($)", min_value=0.0, value=30000.0, step=1000.0)
            MedInc = MedInc_dollars / 10_000
            HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
            AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
            AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
        with col2:
            Population = st.number_input("Population", min_value=0.0, value=1000.0)
            AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
            Country = st.selectbox("Country", countries)
            State = st.selectbox("State", states_by_country.get(Country, []))
            City = st.selectbox("City", cities_by_state.get((Country, State), []))

    if st.button("Predict Price", use_container_width=True):
        default_lat = 34.0
        default_long = -118.0

        input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, default_lat, default_long]],
                                   columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
        prediction = model.predict(input_data)[0]
        predicted_price = prediction * 100_000
        st.success(f"🏠 Predicted House Price: ${predicted_price:,.2f}")

        report_df = pd.DataFrame({
            "Median Income ($)": [MedInc_dollars],
            "House Age": [HouseAge],
            "Average Rooms": [AveRooms],
            "Average Bedrooms": [AveBedrms],
            "Population": [Population],
            "Average Occupancy": [AveOccup],
            "Country": [Country],
            "State": [State],
            "City": [City],
            "Predicted Price ($)": [predicted_price]
        })

        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("📅 Download Prediction Report", data=csv, file_name="prediction_report.csv", mime='text/csv')

        history_file = os.path.join(HISTORY_DIR, f"{st.session_state['login_user']}_history.csv")
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            report_df = pd.concat([history_df, report_df], ignore_index=True)
        report_df.to_csv(history_file, index=False)

def show_bulk_upload_page(model):
    st.subheader("📁 Upload Dataset for Bulk Prediction")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            required_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

            if all(col in user_df.columns for col in required_cols):
                user_df['Latitude'] = 34.0
                user_df['Longitude'] = -118.0

                predictions = model.predict(user_df[required_cols + ['Latitude', 'Longitude']])
                user_df["Predicted Price ($)"] = predictions * 100_000
                st.success("✅ Predictions completed.")
                st.dataframe(user_df)

                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("📅 Download Predictions", data=csv, file_name="bulk_predictions.csv", mime='text/csv')
            else:
                st.error("❌ Uploaded file does not contain all required columns.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

def show_history_page():
    st.subheader("🔓 My Prediction History")
    history_file = os.path.join(HISTORY_DIR, f"{st.session_state['login_user']}_history.csv")
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        st.dataframe(history_df)
    else:
        st.info("No history found yet.")

def show_dataset_info_page(df):
    st.subheader("📈 Dataset Overview")
    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df.head())

    if st.checkbox("Show Correlation Heatmap"):
        correlation = df.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True,
                    annot_kws={'size': 8}, cmap='Blues', ax=ax)
        st.pyplot(fig)

# -------------------------
# Run App
# -------------------------
if st.session_state['logged_in']:
    model, df, X_train, y_train, X_test, y_test = load_model_and_data()

    st.sidebar.title("🔧 Navigation")
    choice = st.sidebar.radio("Go to", ["Predict", "Bulk Upload", "History", "Dataset Info"])
    st.sidebar.markdown("""---""")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state.pop('login_user', None)
        st.rerun()

    st.markdown("""<style>.block-container { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

    if choice == "Predict":
        show_prediction_page(model)
    elif choice == "Bulk Upload":
        show_bulk_upload_page(model)
    elif choice == "History":
        show_history_page()
    elif choice == "Dataset Info":
        show_dataset_info_page(df)
else:
    show_login_register()