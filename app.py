"""
🔧 NASA Turbofan Jet Engine - Predictive Maintenance System
Author: MLOps Practical Project
Description: Streamlit app for predicting Remaining Useful Life (RUL) of turbofan engines
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Page configuration
st.set_page_config(
    page_title="Turbofan Engine RUL Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F44336;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and related files
@st.cache_resource
def load_model_components():
    """Load the Random Forest model, scaler, and model info"""
    try:
        # Load Random Forest model
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler = joblib.load('rf_scaler.pkl')
        
        # Load model info
        with open('rf_model_info.json', 'r') as f:
            model_info = json.load(f)
        
        return model, scaler, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load data
@st.cache_data
def load_data():
    """Load training and test datasets"""
    try:
        # Define file paths
        train_path = 'CMaps/train_FD001.txt'
        test_path = 'CMaps/test_FD001.txt'
        rul_path = 'CMaps/RUL_FD001.txt'
        
        # Load the data
        train_df = pd.read_csv(train_path, sep=' ', header=None)
        test_df = pd.read_csv(test_path, sep=' ', header=None)
        truth_df = pd.read_csv(rul_path, sep=' ', header=None)
        
        # Clean data
        train_df.drop(columns=[26, 27], inplace=True, errors='ignore')
        test_df.drop(columns=[26, 27], inplace=True, errors='ignore')
        truth_df.drop(columns=[1], inplace=True, errors='ignore')
        
        # Define column names
        column_names = [
            'engine_id', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
            'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
            'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
            'sensor_19', 'sensor_20', 'sensor_21'
        ]
        
        train_df.columns = column_names
        test_df.columns = column_names
        truth_df.columns = ['RUL']
        
        # Calculate RUL for training data
        max_cycles_df = train_df.groupby('engine_id')['time_in_cycles'].max().reset_index()
        max_cycles_df.columns = ['engine_id', 'max_cycles']
        train_df = pd.merge(train_df, max_cycles_df, on='engine_id', how='left')
        train_df['RUL'] = train_df['max_cycles'] - train_df['time_in_cycles']
        train_df.drop(columns=['max_cycles'], inplace=True)
        
        return train_df, test_df, truth_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def get_risk_level(rul):
    """Determine risk level based on RUL"""
    if rul <= 25:
        return "🔴 CRITICAL", "danger-box"
    elif rul <= 50:
        return "🟠 WARNING", "warning-box"
    elif rul <= 100:
        return "🟡 CAUTION", "warning-box"
    else:
        return "🟢 HEALTHY", "info-box"

def get_maintenance_recommendation(rul):
    """Provide maintenance recommendation based on RUL with simple explanations"""
    if rul <= 25:
        return """
        #### 🚨 IMMEDIATE ACTION REQUIRED
        
        **What to do RIGHT NOW:**
        - ❌ **Ground this aircraft** - do not fly until serviced
        - 🔧 **Emergency maintenance** within 24-48 hours
        - 👨‍🔧 **Alert maintenance crew** immediately
        
        **Why:** Engine is in critical condition. Like a car with a cracked engine block - could fail any moment.
        
        **Expected:** Full inspection (8-12 hours), likely part replacement, 2-5 days downtime
        
        💡 **Analogy:** Your car's check engine light is flashing. Pull over NOW!
        """
    elif rul <= 50:
        return """
        #### ⚠️ URGENT MAINTENANCE NEEDED
        
        **What to do THIS WEEK:**
        - 📅 **Schedule maintenance** within 5-7 days
        - 🔍 **Increase monitoring** after every flight
        - 🛠️ **Prepare parts** - order components now
        
        **Why:** Engine degrading quickly. Like worn brake pads - still works but you're pushing your luck.
        
        **Expected:** Detailed inspection (4-6 hours), possible replacements, 1-3 days downtime
        
        💡 **Analogy:** Your car is making unusual noises. Drive to the mechanic, but don't take a road trip!
        """
    elif rul <= 100:
        return """
        #### ⚡ PLAN AHEAD FOR MAINTENANCE
        
        **What to do THIS MONTH:**
        - 📆 **Schedule maintenance** within 2-4 weeks
        - 📊 **Monitor trends** - watch for changes
        - 🔧 **Normal operations** continue
        
        **Why:** Engine showing early wear. Like approaching your 5,000-mile oil change - not critical yet.
        
        **Expected:** Standard maintenance (2-4 hours), routine parts, 0.5-1 day downtime
        
        💡 **Analogy:** Your car is due for an oil change soon. Schedule it while you remember!
        """
    else:
        return """
        #### ✅ HEALTHY - ROUTINE MONITORING
        
        **What to do:**
        - 👀 **Continue normal operations** - engine excellent
        - 📊 **Regular monitoring** - standard checks
        - 📅 **Routine inspection** in 3-6 months
        
        **Why:** Engine performing well. Like a car fresh from service - enjoy the smooth ride!
        
        **Expected:** Next routine check in 3-6 months, no downtime
        
        💡 **Analogy:** Your car just passed inspection with flying colors. Just keep up regular maintenance!
        """

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">✈️ NASA Turbofan Engine Predictive Maintenance</div>', unsafe_allow_html=True)
    
    # Load components
    model, scaler, model_info = load_model_components()
    train_df, test_df, truth_df = load_data()
    
    if model is None or train_df is None:
        st.error("Failed to load model or data. Please check if all files are present.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/jet-engine.png", width=100)
        st.title("🎛️ Navigation")
        
        # Beginner tip
        st.success("**🆕 First time?**\n\nStart with 🏠 Home, then try 🎯 Live Predictions!")
        
        page = st.radio(
            "Select a page:",
            ["🏠 Home", "📊 Dataset Overview", "🔍 Exploratory Analysis", 
             "🎯 Live Predictions", "📈 Model Performance", "ℹ️ About"]
        )
        
        st.markdown("---")
        
        # Quick glossary
        with st.expander("📖 Quick Terms", expanded=False):
            st.markdown("""
            **RUL**: Remaining Useful Life  
            How many flights left before service
            
            **Cycle**: One takeoff + landing  
            Most planes: 2-4 cycles/day
            
            **Sensor**: Measurement device  
            Monitors temp, pressure, speed
            
            **AI Model**: Smart predictor  
            Learns from 20,000+ examples
            """)
        
        st.markdown("---")
        st.markdown("### 🔧 Model Stats")
        st.info(f"**Type**: {model_info['model_type']}")
        st.info(f"**Sensors**: {len(model_info['feature_cols'])}")
        st.info(f"**Accuracy**: {model_info['model_performance']['test_r2']:.1%}")
        
        st.markdown("---")
        
        # Color legend
        with st.expander("🎨 Color Guide", expanded=False):
            st.markdown("""
            **Risk Levels:**
            
            🟢 **Healthy**  
            RUL > 100 cycles
            
            🟡 **Caution**  
            RUL 51-100 cycles
            
            🟠 **Warning**  
            RUL 26-50 cycles
            
            🔴 **Critical**  
            RUL ≤ 25 cycles
            """)
        
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Dataset Overview":
        show_dataset_page(train_df, test_df, truth_df)
    elif page == "🔍 Exploratory Analysis":
        show_eda_page(train_df)
    elif page == "🎯 Live Predictions":
        show_prediction_page(model, scaler, model_info, test_df, truth_df)
    elif page == "📈 Model Performance":
        show_performance_page(model_info, test_df, truth_df, model, scaler)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page with simple, friendly introduction"""
    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
        <h2 style='margin:0; color: white;'>🛩️ Predict When Your Engine Needs Maintenance</h2>
        <p style='font-size: 1.2rem; margin-top: 1rem; color: white;'>
            No engineering degree required! We use AI to predict aircraft engine maintenance - before breakdowns happen.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # What is this tool?
    st.markdown("## 🤔 What Does This Tool Do?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 The Simple Answer
        
        This tool predicts **when an aircraft engine will need maintenance** by analyzing sensor data.
        
        **Think of it like this:**
        - Your car tells you when to change oil (every 5,000 miles)
        - This tool tells airlines when to service engines (before problems happen)
        
        **Why is this important?**
        - ✅ Prevents unexpected breakdowns mid-flight
        - ✅ Saves millions in repair costs
        - ✅ Keeps passengers safe
        - ✅ Reduces flight delays
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 The Technical Answer
        
        This is a **Machine Learning** system that:
        
        1. **Learns** from 20,000+ historical engine records
        2. **Analyzes** 17 different sensor measurements
        3. **Predicts** Remaining Useful Life (RUL)
        4. **Recommends** when to schedule maintenance
        
        **Technology Used:**
        - 🤖 Random Forest AI Model
        - 📊 NASA Turbofan Engine Dataset
        - 🎯 81% Prediction Accuracy
        """)
    
    st.markdown("---")
    
    # Visual workflow
    st.markdown("## 🚀 How It Works (In 60 Seconds)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;'>
            <h1 style='margin:0;'>📊</h1>
            <h4 style='color:black'>Step 1</h4>
            <p style='color:black'>Sensors collect data during flight</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;'>
            <h1 style='margin:0;'>💻</h1>
            <h4 style='color:black'>Step 2</h4>
            <p style='color:black'>AI analyzes sensor readings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;'>
            <h1 style='margin:0;'>🔮</h1>
            <h4 style='color:black'>Step 3</h4>
            <p style='color:black'>Predicts remaining cycles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;'>
            <h1 style='margin:0;'>🔧</h1>
            <h4 style='color:black'>Step 4</h4>
            <p style='color:black'>Suggests maintenance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start
    st.markdown("## 🎮 Quick Start Guide")
    
    st.success("""
    ### � Never used this before? Start here!
    
    **Option 1: Try a Demo (Recommended for beginners)**
    1. Click **"🎯 Live Predictions"** in the sidebar
    2. Click the **🟢 Green "Healthy Engine"** button
    3. Click **"🔮 Predict Remaining Useful Life"**
    4. Read the results - we explain everything in plain English!
    
    **Option 2: Explore the Data**
    1. Click **"📊 Dataset Overview"** to see real engine data
    2. Click **"🔍 Exploratory Analysis"** to view charts
    
    **Option 3: Check Model Performance**
    1. Click **"📈 Model Performance"** to see AI accuracy
    """)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Live Predictions
        - Get instant predictions
        - 4 pre-loaded examples
        - Color-coded risk levels
        - Plain English explanations
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data Explorer
        - View 20,000+ engine records
        - Interactive charts
        - Filter by engine ID
        - Download data as CSV
        """)
    
    with col3:
        st.markdown("""
        ### 📈 AI Performance
        - See model accuracy
        - Compare predictions vs reality
        - Understand AI confidence
        - Visualize error rates
        """)
    
    st.markdown("---")
    
    # FAQ
    with st.expander("❓ Frequently Asked Questions", expanded=False):
        st.markdown("""
        ### 🤔 Common Questions
        
        **Q: Do I need aviation knowledge to use this?**  
        A: No! We explain everything simply. If you can check your car's oil, you can use this.
        
        **Q: What is "Remaining Useful Life" (RUL)?**  
        A: How many more flight cycles (takeoffs + landings) before the engine needs service.
        
        **Q: What's a "flight cycle"?**  
        A: One takeoff + one landing = 1 cycle. Most planes fly 2-4 cycles per day.
        
        **Q: How accurate is this?**  
        A: Our AI is 81% accurate on average, trained on real NASA data.
        
        **Q: What if I don't know the sensor values?**  
        A: Just click a colored button (🟢🟡🟠🔴) - we'll load realistic examples!
        
        **Q: Can I upload my own data?**  
        A: Yes! Go to "Live Predictions" and use the "Upload CSV" option.
        """)
    
    st.markdown("---")
    
    # Call to action
    st.info("""
    ### 🚀 Ready to Get Started?
    
    Click **"🎯 Live Predictions"** in the sidebar to try your first prediction!
    
    **Remember:** Start with the 🟢 Green button to see a healthy engine example.
    """)

def show_dataset_page(train_df, test_df, truth_df):
    """Display dataset overview page"""
    st.markdown("## 📊 Dataset Overview")
    
    tab1, tab2, tab3 = st.tabs(["🚂 Training Data", "🧪 Test Data", "📋 Summary Statistics"])
    
    with tab1:
        st.markdown("### Training Dataset")
        st.info(f"**Shape**: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
        
        # Display sample
        st.markdown("#### Sample Data (First 10 rows)")
        st.dataframe(train_df.head(10), use_container_width=True)
        
        # Column info
        st.markdown("#### Dataset Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Identification & Time**")
            st.code("• engine_id: Unique engine identifier\n• time_in_cycles: Operational cycle number")
            
            st.markdown("**Operational Settings**")
            st.code("• setting_1, setting_2, setting_3\n  (Altitude, Mach, TRA)")
        
        with col2:
            st.markdown("**Sensor Measurements (21 sensors)**")
            st.code("""• sensor_1 to sensor_21
  (Temperature, Pressure, Speed, etc.)""")
            
            st.markdown("**Target Variable**")
            st.code("• RUL: Remaining Useful Life (cycles)")
        
        # Engine lifecycle visualization
        st.markdown("#### Engine Lifecycle Example")
        sample_engine = st.selectbox("Select Engine ID", sorted(train_df['engine_id'].unique())[:10], key="train_engine")
        
        engine_data = train_df[train_df['engine_id'] == sample_engine]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=engine_data['time_in_cycles'],
            y=engine_data['RUL'],
            mode='lines+markers',
            name='RUL',
            line=dict(color='#1E88E5', width=3),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.2)'
        ))
        
        fig.update_layout(
            title=f"Engine {sample_engine} - Remaining Useful Life Over Time",
            xaxis_title="Time in Cycles",
            yaxis_title="Remaining Useful Life (cycles)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Engine {sample_engine} Statistics:**
        - Total Operational Cycles: **{engine_data['time_in_cycles'].max()}**
        - Starting RUL: **{engine_data['RUL'].max()}**
        - Final RUL: **{engine_data['RUL'].min()}**
        """)
    
    with tab2:
        st.markdown("### Test Dataset")
        st.info(f"**Shape**: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
        
        st.markdown("#### Sample Data (First 10 rows)")
        st.dataframe(test_df.head(10), use_container_width=True)
        
        st.markdown("#### Ground Truth RUL Values")
        st.info(f"**Shape**: {truth_df.shape[0]} engines")
        
        # Visualize RUL distribution
        fig = px.histogram(
            truth_df, 
            x='RUL', 
            nbins=30,
            title="Distribution of True RUL Values in Test Set",
            labels={'RUL': 'Remaining Useful Life (cycles)'},
            color_discrete_sequence=['#764ba2']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean RUL", f"{truth_df['RUL'].mean():.1f} cycles")
        with col2:
            st.metric("Min RUL", f"{truth_df['RUL'].min()} cycles")
        with col3:
            st.metric("Max RUL", f"{truth_df['RUL'].max()} cycles")
    
    with tab3:
        st.markdown("### Summary Statistics")
        
        # Training data statistics
        st.markdown("#### Training Data Statistics")
        st.dataframe(train_df.describe().transpose(), use_container_width=True)
        
        # Missing values
        st.markdown("#### Data Quality Check")
        missing_train = train_df.isnull().sum().sum()
        missing_test = test_df.isnull().sum().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            if missing_train == 0:
                st.success(f"✅ Training Data: **No missing values** ({train_df.shape[0]:,} rows)")
            else:
                st.warning(f"⚠️ Training Data: **{missing_train} missing values**")
        
        with col2:
            if missing_test == 0:
                st.success(f"✅ Test Data: **No missing values** ({test_df.shape[0]:,} rows)")
            else:
                st.warning(f"⚠️ Test Data: **{missing_test} missing values**")

def show_eda_page(train_df):
    """Display exploratory data analysis page"""
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Sensor Trends", "🔗 Correlations", "📊 Distributions"])
    
    with tab1:
        st.markdown("### Sensor Measurements Over Engine Lifecycle")
        
        # Select engine
        engine_id = st.selectbox("Select Engine ID", sorted(train_df['engine_id'].unique())[:20], key="eda_engine")
        engine_data = train_df[train_df['engine_id'] == engine_id].sort_values('time_in_cycles')
        
        # Select sensors to visualize
        sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
        selected_sensors = st.multiselect(
            "Select Sensors to Visualize",
            sensor_cols,
            default=['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']
        )
        
        if selected_sensors:
            # Create subplot
            fig = make_subplots(
                rows=len(selected_sensors), 
                cols=1,
                subplot_titles=selected_sensors,
                vertical_spacing=0.05
            )
            
            colors = px.colors.qualitative.Set2
            
            for idx, sensor in enumerate(selected_sensors):
                fig.add_trace(
                    go.Scatter(
                        x=engine_data['time_in_cycles'],
                        y=engine_data[sensor],
                        mode='lines',
                        name=sensor,
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ),
                    row=idx+1, col=1
                )
                
                fig.update_xaxes(title_text="Time in Cycles", row=idx+1, col=1)
                fig.update_yaxes(title_text="Value", row=idx+1, col=1)
            
            fig.update_layout(
                height=200 * len(selected_sensors),
                showlegend=False,
                title_text=f"Engine {engine_id} - Sensor Degradation Patterns"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one sensor to visualize")
        
        # RUL degradation
        st.markdown("### RUL Degradation Pattern")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=engine_data['time_in_cycles'],
            y=engine_data['RUL'],
            mode='lines+markers',
            name='RUL',
            line=dict(color='#F44336', width=3),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"Engine {engine_id} - Remaining Useful Life Degradation",
            xaxis_title="Time in Cycles",
            yaxis_title="Remaining Useful Life (cycles)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Feature Correlation Analysis")
        
        # Calculate correlation with RUL
        numeric_cols = [col for col in train_df.columns if col not in ['engine_id', 'time_in_cycles']]
        correlations = train_df[numeric_cols].corr()['RUL'].sort_values(ascending=False)
        
        # Remove RUL itself
        correlations = correlations.drop('RUL')
        
        # Create bar chart
        fig = go.Figure()
        
        colors = ['#4CAF50' if x > 0 else '#F44336' for x in correlations.values]
        
        fig.add_trace(go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{val:.3f}" for val in correlations.values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Correlation of Features with RUL",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Feature",
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - 🟢 **Positive correlation**: Feature increases as RUL increases
        - 🔴 **Negative correlation**: Feature decreases as RUL increases (degradation indicator)
        - Features with higher absolute correlation are more predictive
        """)
    
    with tab3:
        st.markdown("### Feature Distributions")
        
        # Select feature to visualize
        all_features = [col for col in train_df.columns if col not in ['engine_id', 'time_in_cycles', 'RUL']]
        selected_feature = st.selectbox("Select Feature", all_features, key="dist_feature")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                train_df,
                x=selected_feature,
                nbins=50,
                title=f"Distribution of {selected_feature}",
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                train_df,
                y=selected_feature,
                title=f"Box Plot of {selected_feature}",
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown(f"#### Statistics for {selected_feature}")
        stats_df = train_df[selected_feature].describe().to_frame()
        stats_df.columns = ['Value']
        st.dataframe(stats_df, use_container_width=True)

def show_prediction_page(model, scaler, model_info, test_df, truth_df):
    """Display prediction page"""
    st.markdown("## 🎯 Live Predictions")
    
    # Simplified explanation for non-technical users
    st.success("""
    ### 👋 Welcome to the Engine Health Predictor!
    
    **What does this tool do?**  
    This tool predicts how many more flight cycles your aircraft engine can safely operate before needing maintenance.
    
    **Easy as 1-2-3:**
    1. 🎮 **Click a colored button** below to load example engine data (or enter your own)
    2. 🔮 **Click "Predict"** to get results instantly
    3. 📋 **Read the recommendation** - we'll tell you what to do next!
    
    💡 **Tip**: Start with the 🟢 Green button to see a healthy engine example!
    """)
    
    # Sensor information dictionary
    sensor_info = {
        'setting_1': {
            'name': 'Altitude / Flight Condition',
            'unit': 'normalized',
            'range': '0.0 - 0.0042',
            'description': 'Flight altitude setting. Higher values indicate higher altitude operations. Sea level ≈ 0.0, Cruise ≈ 0.0020-0.0042'
        },
        'setting_2': {
            'name': 'Mach Number',
            'unit': 'normalized',
            'range': '0.0 - 0.0014',
            'description': 'Aircraft speed relative to speed of sound. Takeoff ≈ 0.0003, Cruise ≈ 0.0008-0.0014'
        },
        'sensor_2': {
            'name': 'Total Temperature at Fan Inlet (T2)',
            'unit': '°R (Rankine)',
            'range': '518 - 645',
            'description': 'Temperature of air entering the engine fan. Indicates ambient conditions and inlet air temperature. Normal: 518-642°R (58-182°F)'
        },
        'sensor_3': {
            'name': 'Total Temperature at LPC Outlet (T24)',
            'unit': '°R (Rankine)',
            'range': '1580 - 1605',
            'description': 'Temperature after Low Pressure Compressor. Shows compression heating effect. Normal: 1589-1602°R. Higher values indicate degradation.'
        },
        'sensor_4': {
            'name': 'Total Temperature at HPC Outlet (T30)',
            'unit': '°R (Rankine)',
            'range': '1395 - 1430',
            'description': 'Temperature after High Pressure Compressor. Critical indicator of compression efficiency. Normal: 1400-1425°R'
        },
        'sensor_6': {
            'name': 'Pressure at Fan Inlet (P2)',
            'unit': 'psia',
            'range': '14.6 - 14.7',
            'description': 'Ambient air pressure at engine inlet. Typically near standard atmospheric pressure (14.7 psia at sea level)'
        },
        'sensor_7': {
            'name': 'Total Pressure in Bypass Duct (P15)',
            'unit': 'psia',
            'range': '21.6 - 22.3',
            'description': 'Pressure of air bypassing the engine core. Indicates bypass flow health. Normal: 21.6-22.1 psia'
        },
        'sensor_8': {
            'name': 'Total Pressure at HPC Outlet (P30)',
            'unit': 'psia',
            'range': '553 - 557',
            'description': 'High pressure after compression. Key indicator of compressor performance. Normal: 554-556 psia. Increases with degradation.'
        },
        'sensor_9': {
            'name': 'Physical Fan Speed (Nf)',
            'unit': 'rpm',
            'range': '2380 - 2390',
            'description': 'Rotational speed of the fan shaft. Normal operation: 2388 rpm. Critical for thrust generation.'
        },
        'sensor_11': {
            'name': 'Physical Core Speed (Nc)',
            'unit': 'rpm',
            'range': '9040 - 9070',
            'description': 'Rotational speed of compressor/turbine core. Normal: 9046-9060 rpm. Higher speeds indicate higher power demand.'
        },
        'sensor_12': {
            'name': 'Engine Pressure Ratio (epr)',
            'unit': 'ratio',
            'range': '1.18 - 1.37',
            'description': 'Ratio of exhaust to inlet pressure. Direct thrust indicator. Normal: 1.21-1.30. Higher = more thrust.'
        },
        'sensor_13': {
            'name': 'Static Pressure at HPC Outlet (Ps30)',
            'unit': 'psia',
            'range': '47.2 - 48.8',
            'description': 'Static (non-dynamic) pressure at compressor outlet. Indicates compression efficiency. Normal: 47.3-48.5 psia'
        },
        'sensor_14': {
            'name': 'Corrected Fan Speed (NRf)',
            'unit': 'rpm',
            'range': '2385 - 2390',
            'description': 'Fan speed corrected for atmospheric conditions. Used for performance comparison. Normal: 2387-2389 rpm'
        },
        'sensor_15': {
            'name': 'Corrected Core Speed (NRc)',
            'unit': 'rpm',
            'range': '8130 - 8175',
            'description': 'Core speed corrected for temperature and pressure. Standardized performance metric. Normal: 8140-8165 rpm'
        },
        'sensor_17': {
            'name': 'Bleed Enthalpy',
            'unit': 'normalized',
            'range': '390 - 396',
            'description': 'Energy content of air bled from compressor. Used for aircraft systems. Normal: 391-395. Increases with degradation.'
        },
        'sensor_20': {
            'name': 'Demanded Fan Speed',
            'unit': 'rpm',
            'range': '2385 - 2390',
            'description': 'Target fan speed from engine control system. Shows what controller is requesting. Normal: 2388 rpm'
        },
        'sensor_21': {
            'name': 'Demanded Corrected Fan Speed',
            'unit': '%',
            'range': '100.0 - 100.1',
            'description': 'Corrected fan speed demand as percentage of design point. Normal operation at 100%. Deviations indicate off-design operation.'
        }
    }
    
    # Example scenarios
    example_scenarios = {
        '🟢 Healthy Engine (RUL ~150)': {
            'setting_1': 0.0007, 'setting_2': 0.0003,
            'sensor_2': 642.51, 'sensor_3': 1589.70, 'sensor_4': 1400.26,
            'sensor_6': 14.62, 'sensor_7': 21.61, 'sensor_8': 554.36,
            'sensor_9': 2388.06, 'sensor_11': 9046.19, 'sensor_12': 1.30,
            'sensor_13': 47.29, 'sensor_14': 2388.00, 'sensor_15': 8138.62,
            'sensor_17': 392.24, 'sensor_20': 2388.00, 'sensor_21': 100.00
        },
        '🟡 Caution Level (RUL ~75)': {
            'setting_1': 0.0020, 'setting_2': 0.0006,
            'sensor_2': 643.35, 'sensor_3': 1593.26, 'sensor_4': 1408.12,
            'sensor_6': 14.63, 'sensor_7': 21.78, 'sensor_8': 555.02,
            'sensor_9': 2388.04, 'sensor_11': 9051.58, 'sensor_12': 1.32,
            'sensor_13': 47.52, 'sensor_14': 2388.03, 'sensor_15': 8147.28,
            'sensor_17': 393.15, 'sensor_20': 2388.05, 'sensor_21': 100.00
        },
        '🟠 Warning Level (RUL ~40)': {
            'setting_1': 0.0035, 'setting_2': 0.0010,
            'sensor_2': 644.18, 'sensor_3': 1598.44, 'sensor_4': 1417.85,
            'sensor_6': 14.64, 'sensor_7': 21.95, 'sensor_8': 555.78,
            'sensor_9': 2388.02, 'sensor_11': 9058.24, 'sensor_12': 1.34,
            'sensor_13': 47.86, 'sensor_14': 2388.07, 'sensor_15': 8159.45,
            'sensor_17': 394.52, 'sensor_20': 2388.09, 'sensor_21': 100.00
        },
        '🔴 Critical Level (RUL ~15)': {
            'setting_1': 0.0042, 'setting_2': 0.0014,
            'sensor_2': 645.24, 'sensor_3': 1602.76, 'sensor_4': 1425.38,
            'sensor_6': 14.65, 'sensor_7': 22.15, 'sensor_8': 556.72,
            'sensor_9': 2388.01, 'sensor_11': 9065.87, 'sensor_12': 1.36,
            'sensor_13': 48.31, 'sensor_14': 2388.12, 'sensor_15': 8168.92,
            'sensor_17': 395.23, 'sensor_20': 2388.12, 'sensor_21': 100.01
        }
    }
    
    tab1, tab2 = st.tabs(["🎚️ Manual Input", "📁 Batch Prediction"])
    
    with tab1:
        # Add beginner-friendly explanations
        with st.expander("❓ What are these sensors? (Click to learn)", expanded=False):
            st.markdown("""
            ### 🛩️ Understanding Jet Engine Sensors (Simplified)
            
            Think of a jet engine like a car engine, but much more powerful. Just like your car has warning lights and gauges, 
            aircraft engines have sensors that monitor everything.
            
            **What we're measuring:**
            
            🌡️ **Temperature Sensors** - *Like checking if your engine is too hot*
            - Measures how hot different parts of the engine are getting
            - Higher temperatures can mean the engine is working too hard
            
            📊 **Pressure Sensors** - *Like checking tire pressure*
            - Measures air pressure at different points in the engine
            - Tells us if air is flowing properly through the engine
            
            ⚡ **Speed Sensors** - *Like your car's RPM gauge*
            - Measures how fast the engine parts are spinning
            - Faster isn't always better - we need the right speed
            
            📈 **Performance Metrics** - *Like your fuel economy display*
            - Shows how efficiently the engine is running
            - Helps us understand overall engine health
            
            ---
            
            **Don't worry about the numbers!** 
            
            Just click one of the colored buttons below, and we'll fill everything in for you. 
            Each color represents a different engine condition:
            - 🟢 Green = Healthy (like a new car)
            - 🟡 Yellow = Needs attention soon (like needing an oil change)
            - 🟠 Orange = Urgent (like a check engine light)
            - 🔴 Red = Critical (engine needs immediate repair)
            """)
            
            # Create a table with sensor info
            sensor_table = []
            for key in ['setting_1', 'setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 
                       'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
                       'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']:
                if key in sensor_info:
                    info = sensor_info[key]
                    sensor_table.append({
                        'Feature': key,
                        'Name': info['name'],
                        'Unit': info['unit'],
                        'Normal Range': info['range']
                    })
            
            st.dataframe(pd.DataFrame(sensor_table), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Predefined Presets with better explanation
        st.markdown("### 🎮 Choose an Engine Condition")
        
        # Add visual guide
        col_guide1, col_guide2 = st.columns([2, 1])
        with col_guide1:
            st.markdown("""
            **Pick a button to see how our AI predicts engine health:**
            - Click any button to load realistic sensor data
            - See what predictions look like for different engine conditions
            - Learn when maintenance is needed
            """)
        with col_guide2:
            st.info("**New here?**\n\nTry clicking the 🟢 **Healthy Engine** button first!")
        
        cols = st.columns(4)
        for idx, (scenario_name, scenario_values) in enumerate(example_scenarios.items()):
            if cols[idx].button(scenario_name, use_container_width=True, key=f"preset_{idx}"):
                st.session_state['preset_values'] = scenario_values
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📝 Enter Sensor Readings")
        
        feature_cols = model_info['feature_cols']
        
        # Get preset values if loaded
        preset_values = st.session_state.get('preset_values', {})
        
        # Create input form
        with st.form("prediction_form"):
            st.markdown("#### ⚙️ Operational Settings")
            col1, col2 = st.columns(2)
            
            input_values = {}
            
            with col1:
                info = sensor_info['setting_1']
                input_values['setting_1'] = st.number_input(
                    f"{info['name']} [{info['range']}]", 
                    value=float(preset_values.get('setting_1', 0.0007)),
                    format="%.4f",
                    help=f"**{info['name']}**\n\n{info['description']}\n\n**Unit**: {info['unit']}\n**Range**: {info['range']}"
                )
            
            with col2:
                info = sensor_info['setting_2']
                input_values['setting_2'] = st.number_input(
                    f"{info['name']} [{info['range']}]", 
                    value=float(preset_values.get('setting_2', 0.0003)),
                    format="%.4f",
                    help=f"**{info['name']}**\n\n{info['description']}\n\n**Unit**: {info['unit']}\n**Range**: {info['range']}"
                )
            
            st.markdown("#### 🌡️ Temperature Sensors")
            col1, col2, col3 = st.columns(3)
            
            temp_sensors = ['sensor_2', 'sensor_3', 'sensor_4']
            defaults_temp = {'sensor_2': 642.0, 'sensor_3': 1589.0, 'sensor_4': 1400.0}
            for idx, sensor in enumerate(temp_sensors):
                if sensor in feature_cols:
                    info = sensor_info[sensor]
                    with [col1, col2, col3][idx]:
                        input_values[sensor] = st.number_input(
                            f"{info['name']} [{info['range']}]",
                            value=float(preset_values.get(sensor, defaults_temp[sensor])),
                            format="%.2f",
                            help=f"**{info['name']}**\n\n{info['description']}\n\n**Unit**: {info['unit']}\n**Range**: {info['range']}"
                        )
            
            st.markdown("#### 📊 Pressure Sensors")
            col1, col2, col3 = st.columns(3)
            
            pressure_sensors = ['sensor_6', 'sensor_7', 'sensor_8']
            defaults_pressure = {'sensor_6': 14.62, 'sensor_7': 21.61, 'sensor_8': 554.36}
            for idx, sensor in enumerate(pressure_sensors):
                if sensor in feature_cols:
                    info = sensor_info[sensor]
                    with [col1, col2, col3][idx]:
                        input_values[sensor] = st.number_input(
                            f"{info['name']} [{info['range']}]",
                            value=float(preset_values.get(sensor, defaults_pressure[sensor])),
                            format="%.2f",
                            help=f"**{info['name']}**\n\n{info['description']}\n\n**Unit**: {info['unit']}\n**Range**: {info['range']}"
                        )
            
            st.markdown("#### ⚡ Speed Sensors")
            col1, col2 = st.columns(2)
            
            speed_sensors = ['sensor_9', 'sensor_11']
            defaults_speed = {'sensor_9': 2388.0, 'sensor_11': 9046.0}
            for idx, sensor in enumerate(speed_sensors):
                if sensor in feature_cols:
                    info = sensor_info[sensor]
                    with [col1, col2][idx]:
                        input_values[sensor] = st.number_input(
                            f"{info['name']} [{info['range']}]",
                            value=float(preset_values.get(sensor, defaults_speed[sensor])),
                            format="%.2f",
                            help=f"**{info['name']}**\n\n{info['description']}\n\n**Unit**: {info['unit']}\n**Range**: {info['range']}"
                        )
            
            st.markdown("#### 📈 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            perf_sensors = ['sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
            defaults_perf = {
                'sensor_12': 1.30, 'sensor_13': 47.29, 'sensor_14': 2388.0,
                'sensor_15': 8138.0, 'sensor_17': 392.0, 'sensor_20': 2388.0, 'sensor_21': 100.0
            }
            
            for idx, sensor in enumerate(perf_sensors):
                if sensor in feature_cols:
                    info = sensor_info[sensor]
                    with [col1, col2, col3][idx % 3]:
                        input_values[sensor] = st.number_input(
                            f"{info['name']} [{info['range']}]",
                            value=float(preset_values.get(sensor, defaults_perf.get(sensor, 500.0))),
                            format="%.2f",
                            help=f"**{info['name']}**\n\n{info['description']}\n\n**Unit**: {info['unit']}\n**Range**: {info['range']}"
                        )
            
            submit_button = st.form_submit_button("🔮 Predict RUL", use_container_width=True)
        
        if submit_button:
            # Clear preset values after submission
            if 'preset_values' in st.session_state:
                del st.session_state['preset_values']
            
            # Prepare input data
            input_data = []
            for feature in feature_cols:
                input_data.append(input_values.get(feature, 0.0))
            
            input_array = np.array([input_data])
            
            # Scale input
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display results with enhanced explanations
            st.markdown("---")
            st.markdown("## 🎯 Your Prediction Results")
            
            # Add simple explanation banner
            st.info(f"""
            ### 📊 What does this mean?
            
            **Remaining Useful Life (RUL)**: The engine can safely operate for approximately **{prediction:.0f} more flight cycles** 
            before it needs maintenance.
            
            **What's a flight cycle?** One takeoff and landing = 1 cycle. Most aircraft fly 2-4 cycles per day.
            
            **In other words**: This engine has about **{int(prediction * 0.4)} to {int(prediction * 0.5)} days** of safe operation left 
            (assuming 2-3 flights per day).
            """)
            
            # Main prediction display with context
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f'<div class="prediction-box">🔮 {prediction:.0f} Flight Cycles Remaining</div>', 
                           unsafe_allow_html=True)
            
            # Risk assessment with explanation
            risk_level, box_class = get_risk_level(prediction)
            
            # Add emoji and simplified status
            if prediction <= 25:
                status_emoji = "🚨"
                status_text = "**URGENT - Book Maintenance NOW!**"
                urgency = "The engine needs immediate attention - like a car's check engine light that's been on too long."
            elif prediction <= 50:
                status_emoji = "⚠️"
                status_text = "WARNING - Schedule Maintenance Soon"
                urgency = "The engine is showing wear - like tires that are getting low on tread."
            elif prediction <= 100:
                status_emoji = "⚡"
                status_text = "**CAUTION - Plan Ahead**"
                urgency = "The engine is healthy but plan for maintenance - like scheduling your next oil change."
            else:
                status_emoji = "✅"
                status_text = "**HEALTHY - Keep Monitoring**"
                urgency = "The engine is in excellent condition - like a well-maintained car."
            
            st.markdown(f'<div class="{box_class}"><h2  style=color:black>{status_emoji} {risk_level}</h2><p style="font-size:1.2rem; color:black">{status_text}</p></div>', 
                       unsafe_allow_html=True)
            
            # Simple explanation of status
            st.markdown(f"**What this means:** {urgency}")
            
            st.markdown("---")
            
            # Maintenance recommendation with timeline
            st.markdown("### 📋 Recommended Action Plan")
            recommendation = get_maintenance_recommendation(prediction)
            st.markdown(recommendation)
            
            # Add visual timeline
            st.markdown("### 📅 Maintenance Timeline")
            
            # Create progress bar showing urgency
            if prediction > 100:
                progress_color = "normal"
                progress_value = 100
            else:
                progress_value = int((prediction / 100) * 100)
                if prediction <= 25:
                    progress_color = "🔴 Critical Zone"
                elif prediction <= 50:
                    progress_color = "🟠 Warning Zone"
                else:
                    progress_color = "🟡 Caution Zone"
            
            st.progress(progress_value / 100)
            
            if prediction <= 100:
                st.caption(f"Engine health: {progress_value}% of safe operation range")
            else:
                st.caption("Engine health: Excellent condition")
            
            # Additional insights with better labels
            st.markdown("### 📊 Quick Facts")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                days_min = int(prediction * 0.4)
                days_max = int(prediction * 0.5)
                st.metric(
                    "⏰ Time Remaining",
                    f"{days_min}-{days_max} days",
                    help="Estimated days of operation (2-3 flights/day)"
                )
            
            with col2:
                flights = int(prediction)
                st.metric(
                    "✈️ Flights Remaining", 
                    f"~{flights} flights",
                    help="Number of takeoff/landing cycles"
                )
            
            with col3:
                confidence = "High ✅" if prediction > 50 else "Medium ⚠️" if prediction > 25 else "Low ⚠️"
                st.metric("🎯 Confidence", confidence)
            
            with col4:
                priority = "Low" if prediction > 100 else "Medium" if prediction > 50 else "High"
                priority_emoji = "🟢" if prediction > 100 else "🟡" if prediction > 50 else "🔴"
                st.metric("⚡ Priority", f"{priority} {priority_emoji}")
            
            # Add comparison to car maintenance
            st.markdown("---")
            st.markdown("### 🚗 Think of it like Car Maintenance")
            
            if prediction <= 25:
                st.error("**Like:** Your car's engine light is flashing red - don't drive it until fixed!")
            elif prediction <= 50:
                st.warning("**Like:** Your car needs new brake pads soon - schedule service this week.")
            elif prediction <= 100:
                st.info("**Like:** Your car is due for an oil change in 1-2 months - plan ahead.")
            else:
                st.success("**Like:** Your car just had a full service - everything looks great!")
    
    with tab2:
        st.markdown("### Batch Prediction from CSV")
        
        st.markdown("""
        Upload a CSV file with sensor readings. The file should contain the following columns:
        """)
        st.code(", ".join(model_info['feature_cols']))
        
        # Sample CSV download
        sample_data = {}
        for feature in model_info['feature_cols']:
            if 'setting' in feature:
                sample_data[feature] = [0.0, 0.0, 0.0]
            else:
                sample_data[feature] = [500.0, 510.0, 490.0]
        
        sample_df = pd.DataFrame(sample_data)
        
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv,
            file_name="sample_input.csv",
            mime="text/csv"
        )
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                batch_df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Found {len(batch_df)} rows.")
                
                # Display uploaded data
                st.markdown("#### Uploaded Data Preview")
                st.dataframe(batch_df.head(10), use_container_width=True)
                
                # Validate columns
                missing_cols = set(model_info['feature_cols']) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    if st.button("🚀 Run Batch Prediction"):
                        # Prepare data
                        X_batch = batch_df[model_info['feature_cols']].values
                        
                        # Scale
                        X_scaled = scaler.transform(X_batch)
                        
                        # Predict
                        predictions = model.predict(X_scaled)
                        
                        # Add predictions to dataframe
                        results_df = batch_df.copy()
                        results_df['Predicted_RUL'] = predictions
                        results_df['Risk_Level'] = results_df['Predicted_RUL'].apply(
                            lambda x: get_risk_level(x)[0]
                        )
                        
                        # Display results
                        st.markdown("#### Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Average RUL", f"{predictions.mean():.1f}")
                        with col3:
                            critical_count = sum(predictions <= 25)
                            st.metric("Critical Engines", critical_count)
                        with col4:
                            st.metric("Max RUL", f"{predictions.max():.1f}")
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv_results,
                            file_name="rul_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        fig = px.histogram(
                            results_df,
                            x='Predicted_RUL',
                            nbins=30,
                            title="Distribution of Predicted RUL Values",
                            color_discrete_sequence=['#667eea']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

def show_performance_page(model_info, test_df, truth_df, model, scaler):
    """Display model performance page"""
    st.markdown("## 📈 Model Performance Analysis")
    
    # Performance metrics
    st.markdown("### 🎯 Model Performance Metrics")
    
    perf = model_info['model_performance']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Test RMSE",
            f"{perf['test_rmse']:.2f} cycles",
            help="Root Mean Squared Error - Lower is better"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Test R² Score",
            f"{perf['test_r2']:.4f}",
            help="R-squared - Proportion of variance explained (0-1)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "NASA Score",
            f"{perf['test_nasa_score']:.2f}",
            help="NASA asymmetric scoring function - Lower is better"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    
    feature_cols = model_info['feature_cols']
    
    st.info(f"""
    The model uses **{len(feature_cols)} features** selected from the original 24 features:
    - **2 Operational Settings**: setting_1, setting_2
    - **15 Sensor Measurements**: Critical sensors for degradation detection
    
    **Excluded features**: Constant sensors and low-correlation features (sensor_1, sensor_5, sensor_10, sensor_16, etc.)
    """)
    
    # Display features
    st.markdown("#### Selected Features")
    features_df = pd.DataFrame({
        'Feature': feature_cols,
        'Type': ['Setting' if 'setting' in f else 'Sensor' for f in feature_cols]
    })
    st.dataframe(features_df, use_container_width=True)
    
    st.markdown("---")
    
    # Model predictions visualization
    st.markdown("### 📊 Actual vs Predicted RUL")
    
    with st.spinner("Generating predictions on test set..."):
        # Get last measurement for each test engine
        test_last = test_df.groupby('engine_id').last().reset_index()
        
        # Prepare features
        X_test = test_last[feature_cols].values
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_true = truth_df['RUL'].values
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Engine ID': range(1, len(y_true) + 1),
            'Actual RUL': y_true,
            'Predicted RUL': y_pred,
            'Error': y_pred - y_true,
            'Absolute Error': np.abs(y_pred - y_true)
        })
        
        # Scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=np.abs(y_pred - y_true),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Absolute<br>Error")
            ),
            text=[f"Engine {i}<br>Actual: {a:.1f}<br>Predicted: {p:.1f}<br>Error: {e:.1f}" 
                  for i, a, p, e in zip(comparison_df['Engine ID'], y_true, y_pred, y_pred-y_true)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="Actual vs Predicted RUL - Test Set",
            xaxis_title="Actual RUL (cycles)",
            yaxis_title="Predicted RUL (cycles)",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                comparison_df,
                x='Error',
                nbins=30,
                title="Prediction Error Distribution",
                labels={'Error': 'Prediction Error (cycles)'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                comparison_df,
                y='Absolute Error',
                title="Absolute Error Distribution",
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Error statistics
        st.markdown("### 📊 Error Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Error", f"{comparison_df['Error'].mean():.2f}")
        with col2:
            st.metric("Mean Absolute Error", f"{comparison_df['Absolute Error'].mean():.2f}")
        with col3:
            st.metric("Max Overestimate", f"{comparison_df['Error'].max():.2f}")
        with col4:
            st.metric("Max Underestimate", f"{comparison_df['Error'].min():.2f}")
        
        # Detailed results table
        st.markdown("### 📋 Detailed Predictions")
        st.dataframe(
            comparison_df.sort_values('Absolute Error', ascending=False),
            use_container_width=True
        )

def show_about_page():
    """Display about page"""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Predictive Maintenance System** was developed to forecast equipment failures in NASA turbofan 
    jet engines using machine learning techniques. The system analyzes sensor data and operational settings 
    to predict the Remaining Useful Life (RUL) of engines, enabling proactive maintenance decisions.
    
    ### 📊 Dataset: NASA C-MAPSS
    
    The **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset is a benchmark dataset 
    for prognostics research:
    
    - **Publisher**: NASA Ames Prognostics Data Repository
    - **Type**: Simulated turbofan engine run-to-failure data
    - **Engines**: 100 training engines + 100 test engines
    - **Features**: 3 operational settings + 21 sensor measurements
    - **Target**: Remaining Useful Life (RUL) in cycles
    
    **Reference**: Saxena, A., Goebel, K., Simon, D., and Eklund, N., "Damage Propagation Modeling for 
    Aircraft Engine Run-to-Failure Simulation", International Conference on Prognostics and Health 
    Management (PHM08), 2008.
    
    ### 🤖 Machine Learning Model
    
    **Model Type**: Random Forest Regressor
    
    **Architecture**:
    - 100 decision trees
    - Max depth: 10 (regularization)
    - Feature selection: 17 most informative features
    - Preprocessing: MinMax scaling
    
    **Performance**:
    - ✅ Test RMSE: **18.10 cycles**
    - ✅ Test R²: **0.81** (81% variance explained)
    - ✅ NASA Score: **924.41**
    
    ### 🛠️ Technology Stack
    
    - **Python 3.8+**: Core programming language
    - **Scikit-learn**: Machine learning library
    - **Pandas & NumPy**: Data manipulation
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### 💼 Business Value
    
    **Cost Savings**:
    - Reduce unplanned downtime by up to 50%
    - Optimize maintenance scheduling
    - Extend equipment lifespan
    
    **Safety Improvements**:
    - Prevent catastrophic failures
    - Early warning system for critical components
    - Risk-based maintenance prioritization
    
    **Operational Efficiency**:
    - Data-driven decision making
    - Optimized spare parts inventory
    - Improved resource allocation
    
    ### 🌐 Applications
    
    This predictive maintenance approach can be applied to various industries:
    
    - ✈️ **Aviation**: Jet engine maintenance
    - 🏭 **Manufacturing**: Industrial equipment monitoring
    - ⚡ **Energy**: Power generation turbines
    - 🚗 **Automotive**: Vehicle component maintenance
    - 🏥 **Healthcare**: Medical equipment monitoring
    
    ### 📚 How to Use This App
    
    1. **📊 Dataset Overview**: Explore the training and test datasets
    2. **🔍 Exploratory Analysis**: Analyze sensor trends and correlations
    3. **🎯 Live Predictions**: Make RUL predictions with manual input or CSV upload
    4. **📈 Model Performance**: Review model accuracy and predictions
    
    ### 🚀 Future Enhancements
    
    - 🔄 Real-time streaming predictions
    - 📱 Mobile application
    - 🤖 Advanced deep learning models (LSTM, Transformers)
    - 🌐 Multi-dataset support (FD002-FD004)
    - 📊 Advanced anomaly detection
    - 🔔 Automated alerting system
    
    ### 📞 Contact & Support
    
    This project is part of an MLOps practical implementation. For questions or feedback:
    
    - 📧 Email: [your.email@domain.com]
    - 💼 LinkedIn: [Your Profile]
    - 🐙 GitHub: [Your Repository]
    
    ---
    
    ### 📄 License
    
    This project uses the NASA C-MAPSS dataset, which is publicly available for research purposes.
    
    **MIT License** - Feel free to use and modify for educational and research purposes.
    
    ---
    
    ⭐ **Made with ❤️ using Python, Scikit-learn, and Streamlit**
    """)

if __name__ == "__main__":
    main()
