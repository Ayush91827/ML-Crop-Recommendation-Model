import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
from sqlalchemy import create_engine, text
import plotly.express as px
import streamlit.components.v1 as components

DATABASE_URL=st.secrets["postgres"]["url"]

@st.cache_resource()
def get_db_engine():
    return create_engine(DATABASE_URL)

engine=get_db_engine()

@st.cache_resource()
def load_model():
    return joblib.load('trained1stmodel.pkl')

model=load_model()
ranges=model.feature_ranges

#Telemetry Loader
@st.cache_data(ttl=60)
def load_telemetry():
    with engine.connect() as conn:
        return pd.read_sql(
            "SELECT * FROM prediction_telemetry ORDER BY timestamp DESC LIMIT 500;",
            conn
        )

#Logging Function
def log_prediction_to_db(inputs,prediction,latency,status):
    """Inserts prediction record into Cloud PostgreSQL"""
    query=text("""INSERT INTO prediction_telemetry (
            timestamp, predicted_crop, latency_ms, user_status,
            input_n, input_p, input_k, input_temp,
            input_humidity, input_ph, input_rainfall
        ) VALUES (
            :timestamp, :predicted_crop, :latency_ms, :user_status,
            :input_n, :input_p, :input_k, :input_temp,
            :input_humidity, :input_ph, :input_rainfall
        )""")

    with engine.connect() as conn:
        conn.execute(query, {
            "timestamp": datetime.now(),
            "predicted_crop": prediction,
            "latency_ms": latency,
            "user_status": status,
            "input_n": inputs["N"],
            "input_p": inputs["P"],
            "input_k": inputs["K"],
            "input_temp": inputs["temp"],
            "input_humidity": inputs["humidity"],
            "input_ph": inputs["ph"],
            "input_rainfall": inputs["rainfall"]
        })
        conn.commit()

#UI Setup
st.title("Crop Recommender")
st.sidebar.header("Input Soil & Climate Values")

N = st.sidebar.slider("Nitrogen (N)", ranges["N"][0], ranges["N"][1], 90)
P = st.sidebar.slider("Phosphorus (P)", ranges["P"][0], ranges["P"][1], 42)
K = st.sidebar.slider("Potassium (K)", ranges["K"][0], ranges["K"][1], 43)
temperature = st.sidebar.slider("Temperature (°C)",  float(ranges["temperature"][0]), float(ranges["temperature"][1]), 20.0)
humidity = st.sidebar.slider("Humidity (%)", float(ranges["humidity"][0]), float(ranges["humidity"][1]), 82.0)
ph = st.sidebar.slider("Soil pH", float(ranges["ph"][0]), float(ranges["ph"][1]), 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", float(ranges["rainfall"][0]), float(ranges["rainfall"][1]), 202.9)

#N = st.slider("Nitrogen (N)", ranges["N"][0], ranges["N"][1], 90)
#P = st.slider("Phosphorus (P)", ranges["P"][0], ranges["P"][1], 42)
#K = st.slider("Potassium (K)", ranges["K"][0], ranges["K"][1], 43)
#temperature = st.slider("Temperature (°C)", float(ranges["temperature"][0]), float(ranges["temperature"][1]), 20.0)
#humidity = st.slider("Humidity (%)", float(ranges["humidity"][0]), float(ranges["humidity"][1]), 82.0)
#ph = st.slider("Soil pH", float(ranges["ph"][0]), float(ranges["ph"][1]), 6.5)
#rainfall = st.slider("Rainfall (mm)", float(ranges["rainfall"][0]), float(ranges["rainfall"][1]), 202.9)

#Prediction Button
if st.button("Recommend Crop"):
    start_time=time.time()

    inputs={
        "N":N,"P":P,"K":K, "temp":temperature, "humidity":humidity, "ph":ph, "rainfall":rainfall
    }

    try:
        features=[[N,P,K,temperature,humidity,ph,rainfall]]
        prediction=model.predict(features)[0]
        status="Success"
        st.success(f"Recommended Crop:{prediction}")

    except Exception as e:
        prediction="None"
        status="Validation Error"
        st.error(f"Error:{e}")

    latency_ms=round((time.time()-start_time)*1000,2)
    log_prediction_to_db(inputs,prediction,latency_ms,status)

#Telemetry Dashboard
st.subheader("📊 Telemetry Analytics: Predictions by Category")
df=load_telemetry()
if not df.empty:
    crop_counts=df["predicted_crop"].value_counts().reset_index()
    crop_counts.columns=["Crop","Count"]

    fig = px.bar(
        crop_counts,
        x="Count",
        y="Crop",
        orientation="h",
        color="Count",
        color_continuous_scale="Greens",
        title="Total Prediction Inferences by Crop"
    ) 
    fig.update_layout(
        yaxis={"categoryorder":"total ascending"},
        margin=dict(l=20,r=20,t=40,b=20)
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Prediction Latency Over Time")
    latency_fig=px.line(
        df.sort_values("timestamp"),
        x="timestamp",
        y="latency_ms",
        markers=True,
        title="Latency Trend (ms)"
    )
    latency_fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Latency (ms)",
        margin=dict(l=20,r=20,t=40,b=20)
    )
    st.plotly_chart(latency_fig, width='stretch')

else:
    st.info("No telemetry data available yet.")


st.subheader("🔗Exploratory Data Analysis Dashboard Access")
st.markdown("[🌐 Open Full Dashboard](https://public.tableau.com/views/CropRecommendationEDA/PHToleranceDistribution?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)", unsafe_allow_html=True)
tableau_embed_url = "https://public.tableau.com/views/CropRecommendationEDA/PHToleranceDistribution?:showVizHome=no&:embed=true"
components.iframe(tableau_embed_url, height=800, scrolling=True)

st.subheader("🔗Post Model Deployment Dashboard Access")
st.markdown("[🌐 Open Full Dashboard](https://public.tableau.com/views/ModelPredictedPostProductionDatasetAnalysis/InputDriftAnalysis?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)", unsafe_allow_html=True)
tableau_embed_url2= "https://public.tableau.com/views/ModelPredictedPostProductionDatasetAnalysis/InputDriftAnalysis?:showVizHome=no&:embed=true"
components.iframe(tableau_embed_url2, height=800, scrolling=True)

