import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import google.generativeai as genai

warnings.filterwarnings('ignore')

# --- Global Configuration ---
DATA_FILE_PATH = "FetiiAI_Data_Austin.xlsx"
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Use the correct model name here

# --- Page Configuration ---
st.set_page_config(
    page_title="Fetii Advanced AI Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Red & White Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }
    
    .main-header {
        background-color: #D32F2F;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-card {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #D32F2F;
        text-align: center;
    }
    
    .metric-card h3 {
        color: #D32F2F;
        font-size: 1.2em;
    }
    
    .metric-card h2 {
        color: #333333;
        font-size: 2em;
    }

    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 1.2em;
        margin-top: 1em;
        background-color: #F5F5F5;
        border: 1px solid #E0E0E0;
    }
    
    .stButton>button {
        border-radius: 10px;
        border: 1px solid #D32F2F;
        background-color: #D32F2F;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #B71C1C;
        border-color: #B71C1C;
        transform: translateY(-2px);
    }
    
    .prediction-box {
        background: linear-gradient(45deg, #FF5252, #D32F2F);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        border: 2px solid white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background-color: #FFF3F3;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #333333;
        border: 1px solid #FFCDD2;
    }
</style>
""", unsafe_allow_html=True)

# Use st.cache_resource to load heavy data and models once
@st.cache_resource
def load_data_and_train_models(file_path):
    """Loads and preprocesses data from a multi-sheet Excel file."""
    ai_assistant = AdvancedFetiiAI()
    success = ai_assistant.load_data_from_file(file_path)
    return ai_assistant, success

class AdvancedFetiiAI:
    def __init__(self):
        self.data_loaded = False
        self.models = {}
        self.enhanced_data = None
        self.label_encoders = {}
        self.metrics = {}
        self.feature_importances = None
        self.data_sheets = {}
        self.gemini_model = None
        self.api_available = False
        
        # Configure Gemini API
        try:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            self.api_available = True
        except Exception as e:
            st.error(f"Gemini API could not be configured with model '{GEMINI_MODEL_NAME}'. Please check your `.streamlit/secrets.toml` file and the model name. Error: {e}")
            self.api_available = False

    def get_feature_importances(self):
        """Returns the feature importances dataframe."""
        return self.feature_importances

    def load_data_from_file(self, file_path):
        """Loads data from a multi-sheet Excel file."""
        try:
            if not os.path.exists(file_path):
                st.error(f"Data file not found at: {file_path}")
                return False
            
            xls = pd.ExcelFile(file_path)
            self.data_sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
            
            self._process_comprehensive_data()
            self._train_advanced_models()
            
            self.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading or processing data: {e}")
            return False

    def _process_comprehensive_data(self):
        """Merges and processes data from all sheets."""
        trip_data = self.data_sheets.get('Trip Data')
        checked_in_users = self.data_sheets.get('Checked in User ID')
        customer_demographics = self.data_sheets.get('Customer Demographics')
        
        if trip_data is None:
            st.error("Missing 'Trip Data' sheet.")
            return

        trip_data.columns = [re.sub(r'[^a-zA-Z0-9]+', '_', col).lower().strip('_') for col in trip_data.columns]
        if checked_in_users is not None:
            checked_in_users.columns = [re.sub(r'[^a-zA-Z0-9]+', '_', col).lower().strip('_') for col in checked_in_users.columns]
        if customer_demographics is not None:
            customer_demographics.columns = [re.sub(r'[^a-zA-Z0-9]+', '_', col).lower().strip('_') for col in customer_demographics.columns]
            
        df = trip_data.copy()
        if checked_in_users is not None and customer_demographics is not None:
            user_demographics_df = pd.merge(checked_in_users, customer_demographics, on='user_id', how='left')
            aggregated_demographics = user_demographics_df.groupby('trip_id').agg(
                age=('age', 'mean'),
                gender=('gender', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
            ).reset_index()
            df = pd.merge(df, aggregated_demographics, on='trip_id', how='left')

        datetime_cols = [col for col in df.columns if 'date' in col or 'time' in col]
        if datetime_cols:
            main_dt_col = datetime_cols[0]
            df[main_dt_col] = pd.to_datetime(df[main_dt_col], errors='coerce')
            df.dropna(subset=[main_dt_col], inplace=True)
            
            df['hour'] = df[main_dt_col].dt.hour
            df['day_of_week'] = df[main_dt_col].dt.dayofweek
            df['month'] = df[main_dt_col].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
            
        lat_cols = [col for col in df.columns if 'lat' in col]
        lon_cols = [col for col in df.columns if 'lon' in col]
        
        if len(lat_cols) >= 2 and len(lon_cols) >= 2:
            df['trip_distance_km'] = df.apply(
                lambda row: geodesic((row[lat_cols[0]], row[lon_cols[0]]), (row[lat_cols[1]], row[lon_cols[1]])).kilometers
                if pd.notna(row[lat_cols[0]]) and pd.notna(row[lon_cols[0]]) and pd.notna(row[lat_cols[1]]) and pd.notna(row[lon_cols[1]])
                else np.nan,
                axis=1
            )
            
        self.enhanced_data = df.copy()
        self._calculate_advanced_metrics()

    def _calculate_advanced_metrics(self):
        df = self.enhanced_data
        passenger_col = next((col for col in df.columns if 'passenger' in col), None)

        self.metrics = {
            'total_trips': len(df),
            'total_passengers': df[passenger_col].sum() if passenger_col and pd.api.types.is_numeric_dtype(df[passenger_col]) else 'N/A',
            'avg_trip_distance': df['trip_distance_km'].mean() if 'trip_distance_km' in df.columns else 'N/A',
            'avg_passengers_per_trip': df[passenger_col].mean() if passenger_col and pd.api.types.is_numeric_dtype(df[passenger_col]) else 'N/A',
            'peak_hour': df['hour'].mode().iloc[0] if 'hour' in df.columns else 'N/A',
            'weekend_trip_percentage': (df['is_weekend'].sum() / len(df) * 100) if 'is_weekend' in df.columns else 'N/A',
        }
    
    def _train_advanced_models(self):
        df = self.enhanced_data.copy()
        passenger_col = next((col for col in df.columns if 'passenger' in col), None)
        if not passenger_col: return

        df[passenger_col] = pd.to_numeric(df[passenger_col], errors='coerce').fillna(df[passenger_col].median())

        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        features_to_exclude = [passenger_col] + [col for col in df.columns if 'id' in col or df[col].nunique() > 50]
        numerical_features = [f for f in numerical_features if f not in features_to_exclude]
        categorical_features = [f for f in categorical_features if f not in features_to_exclude]

        for col in categorical_features:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        feature_cols = numerical_features + categorical_features
        if not feature_cols: return

        X = df[feature_cols].fillna(0)
        y = df[passenger_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        best_model_name = None
        best_score = -np.inf
        
        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            self.models[name] = {'model': model, 'score': score, 'features': feature_cols}
            if score > best_score:
                best_score = score
                best_model_name = name
        
        if best_model_name:
            self.models['best_model'] = self.models[best_model_name]
            st.session_state.best_model_name = best_model_name
            best_model_obj = self.models['best_model']['model']
            if hasattr(best_model_obj, 'feature_importances_'):
                self.feature_importances = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model_obj.feature_importances_
                }).sort_values('importance', ascending=False)
            
    def predict_passengers(self, features_dict):
        if 'best_model' not in self.models:
            return None, "Prediction model not available."
            
        try:
            model_info = self.models['best_model']
            model = model_info['model']
            feature_cols = model_info['features']
            
            input_df = pd.DataFrame([features_dict])
            
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            categorical_features = [col for col in feature_cols if col in self.label_encoders]
            for col in categorical_features:
                if col in self.label_encoders:
                    input_df[col] = self.label_encoders[col].transform(input_df[col].astype(str))
                else:
                    input_df[col] = 0
            
            input_df = input_df[feature_cols]
            
            prediction = model.predict(input_df)[0]
            confidence = model_info['score']
            
            return max(1, round(prediction)), f"Model: {st.session_state.best_model_name} (R² Score: {confidence:.2f})"
        except Exception as e:
            return None, f"Prediction error: {e}"

    def intelligent_query_processor(self, question):
        question_lower = question.lower().strip()
        
        if 'total trips' in question_lower or 'how many rides' in question_lower:
            return f"📊 **Total Trips**: The dataset contains **{self.metrics.get('total_trips', 'N/A'):,}** trips."
        if 'busiest hour' in question_lower or 'peak time' in question_lower:
            return f"⏰ **Peak Hour**: The busiest hour for trips is **{self.metrics.get('peak_hour', 'N/A')}:00**."
        if 'average passengers' in question_lower or 'avg pax' in question_lower:
            avg_pax = self.metrics.get('avg_passengers_per_trip', 'N/A')
            return f"👥 **Average Passengers**: On average, there are **{avg_pax:.2f}** passengers per trip." if isinstance(avg_pax, (int, float)) else f"👥 **Average Passengers**: N/A"
        
        if self.api_available:
            try:
                system_instruction = "You are a helpful and knowledgeable AI assistant for a ride-sharing company called Fetii. Your task is to answer user questions about the company and its trip data. Use a professional and friendly tone."
                
                response = self.gemini_model.generate_content(
                    f"{system_instruction}\nUser Query: {question}",
                    generation_config=genai.types.GenerationConfig(temperature=0.2)
                )
                
                return f"🤖 **AI Model Response:** {response.text}"
            except Exception as e:
                return f"An error occurred with the Gemini API. Please try a simpler question. Error: {e}"
        else:
            return "The Gemini AI model is not available. Please ensure your API key is correctly set up."
    
def create_analytics_dashboard(ai_assistant):
    st.header("📊 Analytics Dashboard")
    
    if not ai_assistant.data_loaded or ai_assistant.enhanced_data is None:
        st.warning("Data not loaded. Please ensure the 'FetiiAI_Data_Austin.xlsx' file is in the correct path.")
        return
    
    df = ai_assistant.enhanced_data
    
    st.markdown("### Key Metrics")
    m = ai_assistant.metrics
    cols = st.columns(4)
    metric_items = [
        ("Total Trips", f"{m.get('total_trips', 0):,}"),
        ("Total Passengers", f"{m.get('total_passengers', 'N/A'):,}" if isinstance(m.get('total_passengers'), (int, float)) else 'N/A'),
        ("Avg Distance (km)", f"{m.get('avg_trip_distance', 0):.1f}" if isinstance(m.get('avg_trip_distance'), (int, float)) else 'N/A'),
        ("Peak Hour", f"{m.get('peak_hour', 'N/A')}:00" if isinstance(m.get('peak_hour'), (int, float)) else 'N/A')
    ]
    for col, (title, value) in zip(cols, metric_items):
        with col:
            st.markdown(f'<div class="metric-card"><h3>{title}</h3><h2>{value}</h2></div>', unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'hour' in df.columns:
            st.markdown("#### Hourly Trip Patterns")
            hourly_counts = df['hour'].value_counts().sort_index()
            fig = px.bar(x=hourly_counts.index, y=hourly_counts.values, labels={'x': 'Hour of Day', 'y': 'Number of Trips'},
                         color_discrete_sequence=['#D32F2F'])
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        importances = ai_assistant.get_feature_importances()
        if importances is not None and not importances.empty:
            st.markdown("#### Key Predictors for Passenger Count")
            fig = px.bar(importances.head(10), x='importance', y='feature', orientation='h',
                         labels={'importance': 'Importance Score', 'feature': 'Feature'},
                         color_discrete_sequence=['#FF5252'])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available. Ensure a predictive model was successfully trained.")


    with col2:
        if 'day_of_week' in df.columns:
            st.markdown("#### Trips by Day of Week")
            daily_counts = df['day_of_week'].value_counts().sort_index()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_counts.index = [days[i] for i in daily_counts.index]
            fig = px.bar(daily_counts, x=daily_counts.index, y=daily_counts.values,
                         labels={'x': 'Day of Week', 'y': 'Number of Trips'},
                         color_discrete_sequence=['#D32F2F'])
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("#### Feature Correlation Heatmap")
        numerical_df = df.select_dtypes(include=np.number)
        corr = numerical_df.corr()
        fig_heatmap = plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='Reds', linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features')
        st.pyplot(fig_heatmap)

    lat_cols = [col for col in df.columns if 'lat' in col]
    lon_cols = [col for col in df.columns if 'lon' in col]
    if lat_cols and lon_cols:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Geospatial Analysis: Pickup Hotspots")
        pickup_data = df[[lat_cols[0], lon_cols[0]]].dropna()
        if not pickup_data.empty:
            fig = px.density_mapbox(pickup_data, lat=lat_cols[0], lon=lon_cols[0], radius=10,
                                    center=dict(lat=pickup_data[lat_cols[0]].mean(), lon=pickup_data[lon_cols[0]].mean()),
                                    zoom=10, mapbox_style="carto-positron",
                                    color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

def create_prediction_interface(ai_assistant):
    st.header("🔮 Predictive Analytics")
    
    if not ai_assistant.data_loaded or 'best_model' not in ai_assistant.models:
        st.warning("Data not loaded or model not trained. Please ensure the 'FetiiAI_Data_Austin.xlsx' file is in the correct path and contains a 'passenger' column.")
        return
    
    st.markdown("### Forecast Passenger Demand")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hour = st.slider("Hour of Day", 0, 23, 18)
    with col2:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = st.selectbox("Day of Week", days, index=4)
        day_of_week = days.index(day_name)
    with col3:
        month = st.slider("Month", 1, 12, 9)
    
    if st.button("🚀 Generate Prediction", use_container_width=True):
        with st.spinner("Calculating..."):
            features = {
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            }
            
            prediction, confidence = ai_assistant.predict_passengers(features)
            
            if prediction:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Passenger Count</h3>
                    <h1>{prediction}</h1>
                    <p>{confidence}</p>
                    <p>For {day_name} at {hour}:00 in month {month}</p>
                </div>
                """, unsafe_allow_html=True)
                
                insight = generate_business_insight(prediction, hour, day_of_week)
                st.markdown(f'<div class="insight-box"><h3>💡 Business Insight</h3><p>{insight}</p></div>', unsafe_allow_html=True)
            else:
                st.error(f"Prediction failed: {confidence}")

def generate_business_insight(prediction, hour, day_of_week):
    insight = []
    if prediction >= 10:
        insight.append("🔥 **High Demand Alert:** This is a peak time. Ensure maximum vehicle availability and consider implementing surge pricing.")
    elif 4 <= prediction < 10:
        insight.append("⚡ **Moderate Demand:** Expect steady business. Standard operational capacity should be sufficient.")
    else:
        insight.append("📉 **Low Demand:** This is an off-peak period. A good time for vehicle maintenance or offering promotional fares to stimulate demand.")
    
    if day_of_week >= 5 and hour >= 18:
        insight.append("🎉 **Weekend Evening:** High demand is likely concentrated around entertainment districts, restaurants, and event venues.")
    elif day_of_week < 5 and ((7 <= hour <= 9) or (17 <= hour <= 19)):
        insight.append("💼 **Commuter Rush Hour:** Focus on business districts and residential transit hubs.")
    
    return " ".join(insight)

def main():
    st.markdown('<div class="main-header"><h1>🚗 Fetii Advanced AI Assistant</h1><p>Data-driven insights and predictive analytics at your fingertips.</p></div>', unsafe_allow_html=True)
    
    ai_assistant, success = load_data_and_train_models(DATA_FILE_PATH)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.info("Data is pre-loaded from `FetiiAI_Data_Austin.xlsx`")
        
        if success:
            st.success("AI Ready")
            st.header("Quick Actions")
            quick_queries = ["What is the busiest hour?", "How many total trips are there?", "Tell me about Fetii"]
            for query in quick_queries:
                if st.button(query, use_container_width=True):
                    st.session_state.user_query = query
                    st.rerun()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.error("AI is not ready. Check file and API key setup.")
    
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat Assistant", "📊 Analytics Dashboard", "🔮 Predictive Analytics"])
    
    with tab1:
        st.header("💬 Intelligent Chat Assistant")
        st.markdown("Ask me anything about your Fetii trip data!")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
        
        user_query = st.chat_input("Ask a question...")
        
        if 'user_query' in st.session_state:
            user_query = st.session_state.pop('user_query')
        
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    if success:
                        response = ai_assistant.intelligent_query_processor(user_query)
                    else:
                        response = "I'm not ready yet. Please ensure the data file and API key are correctly configured."
                    st.markdown(response, unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    with tab2:
        create_analytics_dashboard(ai_assistant)
    
    with tab3:
        create_prediction_interface(ai_assistant)

if __name__ == "__main__":
    main()


