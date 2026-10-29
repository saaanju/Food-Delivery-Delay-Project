import pickle
import pandas as pd
import streamlit as st

# STEP 1: Page Configuration set to WIDE layout
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling: Purple, White, Dark/Black Theme
st.markdown(
    """
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #0F0E17;
        color: #FFFFF0;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #6C5CE7 0%, #4834D4 100%);
        padding: 1.8rem;
        border-radius: 16px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.3);
        margin-bottom: 2rem;
    }

    .header-box h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }

    .header-box p {
        font-size: 0.95rem;
        color: #E0D9FF;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Side Cards & Containers */
    .info-card {
        background-color: #1A1926;
        border: 1.5px solid #2D2B42;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .info-title {
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #A29BFE;
        margin-bottom: 0.75rem;
    }

    .info-item {
        font-size: 0.85rem;
        color: #DCDDE1;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }

    /* Input Styling */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #1A1926 !important;
        border: 1.5px solid #2D2B42 !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #A29BFE !important;
        box-shadow: 0 0 0 3px rgba(162, 155, 254, 0.25) !important;
    }

    input {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .stNumberInput label, .stSelectbox label {
        color: #DCDDE1 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6C5CE7 0%, #512DA8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        width: 100%;
        margin-top: 0.5rem;
        transition: all 0.25s ease;
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.4);
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.6);
    }

    /* Result Card Styling */
    .result-card {
        background-color: #1A1926;
        border: 2px solid #6C5CE7;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.2);
    }

    .result-value {
        font-size: 3rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.2rem 0;
        line-height: 1.1;
    }

    /* Summary Metrics Table */
    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid #2D2B42;
        color: #DCDDE1;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.75rem;
        color: #71717A;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# STEP 2: Load the model
@st.cache_resource
def load_model():
    with open("delivery_model.pkl", "rb") as file:
        package = pickle.load(file)
    return package["model"], package["features"]


model, features = load_model()

# STEP 3: Header Section
st.markdown(
    """
    <div class="header-box">
        <h1>🍔 Food Delivery Time Predictor</h1>
        <p>Smart Machine Learning Dashboard for Order Estimation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# STEP 4: 3-Column Dashboard Layout
left_col, center_col, right_col = st.columns([1, 1.3, 1], gap="large")

# LEFT COLUMN: Guidance & Model Features
with left_col:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">💡 How It Works</div>
            <div class="info-item">This application uses a trained <b>Linear Regression</b> model to calculate food delivery time based on operational factors.</div>
        </div>
        <div class="info-card">
            <div class="info-title">📊 Key Predictors</div>
            <div class="info-item">📍 <b>Distance:</b> Route length between restaurant and buyer.</div>
            <div class="info-item">🍳 <b>Prep Time:</b> Estimated kitchen cooking time.</div>
            <div class="info-item">🚦 <b>Traffic Level:</b> Real-time road congestion score (1-4).</div>
            <div class="info-item">🌧️ <b>Weather:</b> Adverse climate severity index (0-3).</div>
            <div class="info-item">🏍️ <b>Rider Experience:</b> Years of courier service.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# CENTER COLUMN: Primary Inputs & Action Button
with center_col:
    st.markdown("### 📝 Enter Delivery Details")

    distance_km = st.number_input(
        "Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.5
    )

    preparation_time_min = st.number_input(
        "Prep Time (mins)", min_value=1, max_value=120, value=15
    )

    traffic_level = st.selectbox(
        "Traffic Level",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Low",
            2: "2 - Moderate",
            3: "3 - High",
            4: "4 - Severe",
        }[x],
    )

    weather_severity = st.selectbox(
        "Weather Severity",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Clear",
            1: "1 - Light Adverse",
            2: "2 - Bad Weather",
            3: "3 - Severe Weather",
        }[x],
    )

    rider_experience_years = st.number_input(
        "Rider Experience (years)", min_value=0, max_value=50, value=3
    )

    predict_btn = st.button("Predict Delivery Time", type="primary")

# RIGHT COLUMN: Predictions & Summary Statistics
with right_col:
    st.markdown("### 🎯 Live Prediction")

    # Run Prediction
    new_delivery = pd.DataFrame(
        [
            [
                distance_km,
                preparation_time_min,
                traffic_level,
                weather_severity,
                rider_experience_years,
            ]
        ],
        columns=features,
    )
    prediction = max(0, model.predict(new_delivery)[0])

    # Display Prediction Box
    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size: 0.8rem; font-weight: 800; text-transform: uppercase; color: #A29BFE;">Estimated Delivery Time</div>
            <div class="result-value">{prediction:.0f} <span style="font-size: 1.5rem; font-weight: 600; color: #A29BFE;">mins</span></div>
            <div style="font-size: 0.8rem; color: #A0A0B2;">Algorithmic duration estimate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Summary Box
    st.write("")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">📋 Order Summary</div>
            <div class="metric-row"><span>Route Distance:</span><b>{distance_km} km</b></div>
            <div class="metric-row"><span>Kitchen Prep:</span><b>{preparation_time_min} mins</b></div>
            <div class="metric-row"><span>Traffic Level:</span><b>Level {traffic_level}</b></div>
            <div class="metric-row"><span>Weather Index:</span><b>Index {weather_severity}</b></div>
            <div class="metric-row" style="border: none;"><span>Rider Experience:</span><b>{rider_experience_years} yrs</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    '<div class="footer-text">Educational Project • Python • Pandas • Scikit-Learn • Streamlit</div>',
    unsafe_allow_html=True,
)