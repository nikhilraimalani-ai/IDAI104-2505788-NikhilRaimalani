import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- SETTINGS & THEME ---
st.set_page_config(page_title="AstroCompute Ultra", page_icon="🔭", layout="wide")

# Peak Design CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .main { background-color: #050505; color: #e0e0e0; }
    .stMetric { 
        background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
        padding: 20px; border-radius: 15px; border: 1px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #005f73);
        color: white; border: none; border-radius: 8px;
        font-weight: bold; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px #00d4ff; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 2px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv('space_missions_dataset.csv')
    df['Launch Date'] = pd.to_datetime(df['Launch Date'])
    # Feature Engineering for Insights
    df['Efficiency_Score'] = df['Payload Weight (tons)'] / (df['Fuel Consumption (tons)'] + 1)
    df['Cost_Per_Ton'] = df['Mission Cost (billion USD)'] / df['Payload Weight (tons)']
    return df

df = load_and_preprocess()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2590/2590324.png", width=100)
    st.title("COMMAND CENTER")
    page = st.radio("SELECT MODULE", ["Mission Intelligence", "Orbital Simulator", "Technical Audit"])
    
    st.markdown("---")
    st.subheader("Global Filters")
    vehicles = st.multiselect("Active Fleet", options=df['Launch Vehicle'].unique(), default=df['Launch Vehicle'].unique())
    target_type = st.multiselect("Target Profiles", options=df['Target Type'].unique(), default=df['Target Type'].unique())

# Filter data globally
f_df = df[(df['Launch Vehicle'].isin(vehicles)) & (df['Target Type'].isin(target_type))]

# --- MODULE 1: MISSION INTELLIGENCE ---
if page == "Mission Intelligence":
    st.header("🌌 Mission Intelligence Dashboard")
    
    # ROW 1: METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ANALYZED MISSIONS", len(f_df))
    m2.metric("AVG SUCCESS RATE", f"{f_df['Mission Success (%)'].mean():.1f}%")
    m3.metric("TOTAL SCIENTIFIC YIELD", f"{f_df['Scientific Yield (points)'].sum():,.0f}")
    m4.metric("AVG COST / MISSION", f"${f_df['Mission Cost (billion USD)'].mean():.2f}B")

    # ROW 2: ADVANCED VISUALS
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("3D Mission Mapping (Distance vs Cost vs Yield)")
        fig_3d = px.scatter_3d(
            f_df, x='Distance from Earth (light-years)', y='Mission Cost (billion USD)', 
            z='Scientific Yield (points)', color='Launch Vehicle', 
            size='Payload Weight (tons)', opacity=0.7, template="plotly_dark"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_b:
        st.subheader("Cost Efficiency Distribution")
        fig_violin = px.violin(f_df, y="Cost_Per_Ton", x="Launch Vehicle", color="Launch Vehicle", box=True, template="plotly_dark")
        st.plotly_chart(fig_violin, use_container_width=True)

    # ROW 3: TIME SERIES
    st.subheader("Historical Timeline of Space Exploration")
    timeline_df = f_df.groupby(f_df['Launch Date'].dt.year).agg({'Mission Cost (billion USD)': 'sum', 'Mission ID': 'count'}).reset_index()
    fig_line = px.area(timeline_df, x='Launch Date', y='Mission Cost (billion USD)', title="Annual Investment Trend", template="plotly_dark", line_shape="spline")
    st.plotly_chart(fig_line, use_container_width=True)

# --- MODULE 2: ORBITAL SIMULATOR ---
elif page == "Orbital Simulator":
    st.header("🧪 Advanced Flight Dynamics Lab")
    
    # Image for visual context
    st.markdown("")

    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.write("### Simulation Parameters")
        p_thrust = st.slider("Engine Thrust (kN)", 1000, 20000, 8000)
        p_fuel = st.number_input("Fuel Load (kg)", 10000, 500000, 150000)
        p_payload = st.number_input("Payload Mass (kg)", 1000, 50000, 15000)
        p_drag = st.slider("Drag Coefficient (Cd)", 0.1, 1.0, 0.4)
        
        st.markdown("### Governing Equation")
        st.latex(r"a = \frac{T - (m \cdot g) - (0.5 \cdot \rho \cdot v^2 \cdot C_d \cdot A)}{m}")
        
        launch = st.button("INITIATE LAUNCH SEQUENCE")

    with col_viz:
        if launch:
            # PHYSICS ENGINE
            dt = 1
            g = 9.81
            dry_mass = 20000
            alt, vel, fuel_rem = 0, 0, p_fuel
            history = []
            
            for t in range(300):
                curr_m = dry_mass + p_payload + fuel_rem
                # Dynamic Air Density (Barometric formula simplification)
                rho = 1.225 * np.exp(-alt / 8500)
                
                if fuel_rem > 0:
                    thrust_force = p_thrust * 1000
                    fuel_rem -= 400 * dt # Fixed burn rate
                else:
                    thrust_force = 0
                
                drag_force = 0.5 * rho * (vel**2) * p_drag * 10
                gravity_force = curr_m * g
                
                net_f = thrust_force - gravity_force - drag_force
                acc = net_f / curr_m
                
                vel += acc * dt
                alt += vel * dt
                
                if alt < 0 and t > 0: break
                history.append({"Time": t, "Altitude": alt/1000, "Velocity": vel, "Acceleration": acc})
            
            sim_df = pd.DataFrame(history)
            
            # Multi-chart display
            sub1 = px.line(sim_df, x="Time", y="Altitude", title="Altitude (km)", template="plotly_dark")
            sub1.update_traces(line_color="#00d4ff")
            st.plotly_chart(sub1, use_container_width=True)
            
            sub2 = px.line(sim_df, x="Time", y="Velocity", title="Velocity (m/s)", template="plotly_dark")
            sub2.update_traces(line_color="#ff007b")
            st.plotly_chart(sub2, use_container_width=True)
        else:
            st.info("Adjust parameters and press Launch to compute trajectory.")

# --- MODULE 3: TECHNICAL AUDIT ---
elif page == "Technical Audit":
    st.header("📝 Project Methodology & Data Audit")
    
    st.subheader("Data Quality Report")
    st.write(f"Sample Size: {len(df)} missions | Columns: {len(df.columns)}")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Variable Correlation Analysis")
    c_matrix = df.select_dtypes(include=[np.number]).corr()
    fig_heat = px.imshow(c_matrix, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("""
    ### Technical Implementation Details
    1. **Data Preprocessing:** Handled datetime objects and engineered 'Cost Per Ton' to measure economic efficiency.
    2. **Physics Simulation:** Used iterative Euler integration to solve for motion.
    3. **UI/UX:** Implemented custom CSS to simulate a dark-mode aerospace interface.
    4. **Interactivity:** Integrated Plotly for zero-lag, zoomable data exploration.
    """)
