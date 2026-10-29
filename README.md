# 🍔 Food Delivery Time Predictor

A complete end-to-end Machine Learning project built with Python, Scikit-Learn, and Streamlit to estimate food delivery durations based on real-world operational factors such as route distance, kitchen prep time, traffic conditions, weather severity, and rider experience.

## 📌 Project Overview

Accurate delivery time estimation is critical for customer satisfaction in food delivery platforms. This repository covers the full data science lifecycle:

1. **Synthetic Data Generation**: Creating historical delivery records based on realistic non-linear operational formulas.
2. **Exploratory Data Analysis (EDA)**: Investigating feature distributions, missing values, and correlation with delivery times.
3. **Model Training & Evaluation**: Training a Multiple Linear Regression model and evaluating its performance using MAE, MSE, RMSE, and $R^2$ metrics.
4. **CLI Prediction Script**: Interactive command-line utility for real-time predictions.
5. **Interactive Web Dashboard**: A modern, dark-themed Streamlit web interface for intuitive user interaction.

---

## 🔄 End-to-End Project Workflow

```
               [ 01_create_data.py ]
                         │
                         ▼
                ( delivery_data.csv )
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
[ 02_explore_data.py ]          [ 03_train_model.py ]
(EDA & Visualizations)          (Linear Regression)
                                          │
                                          ▼
                                ( delivery_model.pkl )
                                          │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                        ▼                                   ▼
              [ 04_predict.py ]                       [ 05_app.py ]
               (CLI Interface)                     (Streamlit Web App)
```

### Detailed Pipeline Breakdown

1. **Data Ingestion & Synthesis (`01_create_data.py`)**: 
   - Generates 1,500 synthetic delivery records using controlled uniform and normal random distributions.
   - Outputs `delivery_data.csv`.

2. **Exploration & Analytics (`02_explore_data.py`)**:
   - Performs data health checks (missing values, data types).
   - Computes feature-to-target correlations and outputs scatter plots for analysis.

3. **Model Training & Serialization (`03_train_model.py`)**:
   - Performs an 80/20 train-test split (`random_state=42`).
   - Trains a `LinearRegression` model from `scikit-learn`.
   - Evaluates performance metrics ($R^2$, MAE, RMSE).
   - Serializes the trained model and feature list into `delivery_model.pkl`.

4. **Deployment & Consumption**:
   - **CLI (`04_predict.py`)**: Prompts user inputs via terminal for quick single-instance inference.
   - **Web Application (`05_app.py`)**: Interactive Streamlit dashboard with real-time UI rendering and parameter adjustment.

---

## 📁 Repository Structure

```
.
├── 01_create_data.py   # Generates the synthetic delivery dataset (delivery_data.csv)
├── 02_explore_data.py  # EDA script for statistical summaries and correlation visualization
├── 03_train_model.py   # Trains the Linear Regression model and exports delivery_model.pkl
├── 04_predict.py       # Command-line interface for running individual predictions
├── 05_app.py           # Streamlit web dashboard implementation
├── delivery_data.csv   # Historical delivery dataset (1,500 records)
├── delivery_model.pkl  # Pickled machine learning model and feature mappings
└── README.md           # Project documentation
```

## 📊 Dataset & Features

The model uses 5 primary operational predictor variables to estimate the target duration:

| Feature Name | Type | Unit / Scale | Description | 
| ----- | ----- | ----- | ----- | 
| `distance_km` | Float | Kilometers (1.0 - 15.0) | Distance between restaurant and customer | 
| `preparation_time_min` | Integer | Minutes (5 - 30) | Time required by the kitchen to prepare the order | 
| `traffic_level` | Integer | Scale (1 - 4) | 1: Low, 2: Moderate, 3: High, 4: Severe | 
| `weather_severity` | Integer | Scale (0 - 3) | 0: Clear, 1: Light Adverse, 2: Bad, 3: Severe | 
| `rider_experience_years` | Integer | Years (0 - 10) | Total delivery experience of the courier | 
| **`delivery_time_min`** | Float | **Minutes (Target)** | Total time elapsed from order placement to delivery | 

## 📈 Model Performance & Results

The dataset of 1,500 records was split into an **80% training set (1,200 samples)** and a **20% testing set (300 samples)**.

### Evaluation Metrics

* **MAE (Mean Absolute Error):** 2.51 minutes
* **MSE (Mean Squared Error):** 9.63
* **RMSE (Root Mean Squared Error):** 3.10 minutes
* $R^2$ **Score:** 0.9649 (Explains \~96.5% of variance)

### Learned Formula / Coefficients

$$
\text{Delivery Time} \approx 4.941 + 3.007(\text{distance}) + 0.997(\text{prep\_time}) + 3.980(\text{traffic}) + 1.979(\text{weather}) - 0.358(\text{rider\_exp})
$$

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.9+** installed on your system.

### 1. Installation

Clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/food-delivery-time-predictor.git
cd food-delivery-time-predictor

pip install pandas numpy scikit-learn matplotlib streamlit
```

### 2. Run Pipeline Steps

**Step 1: Generate Dataset**
```bash
python 01_create_data.py
```

**Step 2: Explore Data**
```bash
python 02_explore_data.py
```

**Step 3: Train & Save Model**
```bash
python 03_train_model.py
```

**Step 4: Interactive CLI Prediction**
```bash
python 04_predict.py
```

## 🌐 Running the Web Application

To launch the Streamlit dashboard:

```bash
streamlit run 05_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### App Features:

* **Responsive 3-Column Layout** optimized for desktop view.
* **Custom Dark Theme** (`#0F0E17` background with purple accent highlights).
* **Live Interactive Slider / Number Inputs** for seamless estimation.
* **Real-Time Order Summary Card**.

## 🛠️ Built With

* [**Python**](https://www.python.org/) - Core programming language
* [**Pandas**](https://pandas.pydata.org/) & [**NumPy**](https://numpy.org/) - Data manipulation and mathematical operations
* [**Scikit-Learn**](https://scikit-learn.org/) - Machine Learning (Linear Regression, Model Evaluation, Data Splitting)
* [**Matplotlib**](https://matplotlib.org/) - Data Visualization
* [**Streamlit**](https://streamlit.io/) - Web Application Framework

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
