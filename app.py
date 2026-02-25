import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# --- 1. EMBEDDED DATASET ENGINE ---
@st.cache_data
def load_embedded_data():
    # This is your full dataset converted to a CSV-style string for the code
    csv_data = """Mission ID,Mission Name,Launch Date,Target Type,Target Name,Mission Type,Distance from Earth (light-years),Mission Duration (years),Mission Cost (billion USD),Scientific Yield (points),Crew Size,Mission Success (%),Fuel Consumption (tons),Payload Weight (tons),Launch Vehicle
MSN-0001,Mission-1,2025-01-01,Star,Titan,Colonization,7.05,5.2,526.68,64.3,21,100.0,731.88,99.78,SLS
MSN-0002,Mission-2,2025-01-08,Exoplanet,Betelgeuse,Colonization,41.76,23.0,234.08,84.4,72,89.6,4197.41,45.72,Starship
MSN-0003,Mission-3,2025-01-15,Asteroid,Mars,Exploration,49.22,28.8,218.68,98.6,16,98.6,4908.0,36.12,Starship
MSN-0004,Mission-4,2025-01-22,Exoplanet,Titan,Colonization,26.33,17.8,232.89,36.0,59,90.0,2569.05,40.67,Starship
MSN-0005,Mission-5,2025-01-29,Exoplanet,Proxima b,Mining,8.67,9.2,72.14,96.5,31,73.2,892.76,12.4,Starship
MSN-0006,Mission-6,2025-02-05,Planet,Mars,Exploration,38.2,16.5,412.5,75.0,12,95.0,3100.0,55.0,Falcon Heavy
MSN-0007,Mission-7,2025-02-12,Moon,Luna,Research,0.00004,0.5,15.2,88.0,4,99.0,150.0,5.5,Ariane 6
""" 
    # Note: In your real file, you can paste more rows here or use the read_csv directly.
    # To keep this snippet clean for you, I'll use a small sample, 
    # but I recommend keeping the 'space_missions_dataset.csv' in the folder for the full 500 rows.
    
    # Check if local file exists, otherwise use embedded sample
    try:
        df = pd.read_csv('space_missions_dataset.csv')
    except:
        df = pd.read_csv(io.StringIO(csv_data))
        
    df['Launch Date'] = pd.to_datetime(df['Launch Date'])
    df['Cost_Per_Ton'] = df['Mission Cost (billion USD)'] / df['Payload Weight (tons)']
    return df

# --- 2. PAGE CONFIG & THEME ---
st.set_page_config(page_title="AstroCompute Ultra", page_icon="🔭", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .main { background-color: #050505; color: #e0e0e0; }
    .stMetric { background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%); border: 1px solid #00d4ff; border-radius: 15px; padding: 20px; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #00d4ff !important; }
    .stButton>button { background: linear-gradient(45deg, #00d4ff, #005f73); color: white; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

df = load_embedded_data()

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚀 COMMAND")
    page = st.radio("SELECT MODULE", ["Intelligence", "Physics Lab", "Data Audit"])
    st.markdown("---")
    vehicles = st.multiselect("Active Fleet", options=df['Launch Vehicle'].unique(), default=df['Launch Vehicle'].unique())

f_df = df[df['Launch Vehicle'].isin(vehicles)]

# --- 4. MODULES ---
if page == "Intelligence":
    st.header("🌌 Mission Intelligence")
    m1, m2, m3 = st.columns(3)
    m1.metric("ANALYZED MISSIONS", len(f_df))
    m2.metric("AVG SUCCESS", f"{f_df['Mission Success (%)'].mean():.1f}%")
    m3.metric("AVG COST", f"${f_df['Mission Cost (billion USD)'].mean():.2f}B")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        fig_3d = px.scatter_3d(f_df, x='Distance from Earth (light-years)', y='Mission Cost (billion USD)', 
                               z='Scientific Yield (points)', color='Launch Vehicle', template="plotly_dark")
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_right:
        fig_box = px.box(f_df, x="Launch Vehicle", y="Cost_Per_Ton", color="Launch Vehicle", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

elif page == "Physics Lab":
    st.header("🧪 Orbital Simulator")
    st.latex(r"a = \frac{T - (m \cdot g) - D}{m}")
    
    with st.expander("Configure Launch Parameters"):
        c1, c2, c3 = st.columns(3)
        p_thrust = c1.slider("Thrust (kN)", 1000, 20000, 8000)
        p_fuel = c2.number_input("Fuel (kg)", 10000, 500000, 150000)
        p_payload = c3.number_input("Payload (kg)", 1000, 50000, 15000)
    
    if st.button("INITIATE LAUNCH"):
        dt, g, dry_m, alt, vel, f_rem = 1, 9.81, 20000, 0, 0, p_fuel
        history = []
        for t in range(200):
            curr_m = dry_m + p_payload + f_rem
            rho = 1.225 * np.exp(-alt / 8500) # Atmospheric density
            thrust = (p_thrust * 1000) if f_rem > 0 else 0
            f_rem -= 400 if f_rem > 0 else 0
            acc = (thrust - (curr_m * g) - (0.5 * rho * vel**2 * 0.4 * 10)) / curr_m
            vel += acc * dt
            alt += vel * dt
            if alt < 0 and t > 0: break
            history.append({"Time": t, "Altitude": alt/1000, "Velocity": vel})
        
        sim_df = pd.DataFrame(history)
        st.line_chart(sim_df.set_index("Time")[["Altitude", "Velocity"]])

else:
    st.header("📝 Data Audit")
    st.dataframe(f_df, use_container_width=True)
    st.subheader("Correlation Matrix")
    fig_heat = px.imshow(f_df.select_dtypes(include=[np.number]).corr(), text_auto=True, template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)
