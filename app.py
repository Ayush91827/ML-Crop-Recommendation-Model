import streamlit as st
import pandas as pd
import joblib

if "history" not in st.session_state:
    st.session_state["history"] = []
    
@st.cache_resource()
def load_model():
    return joblib.load('trained1stmodel.pkl')

model=load_model()
ranges=model.feature_ranges

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

if st.button("Recommend Crop"):
    input_df=pd.DataFrame([{
        "N":N,
        "P":P,
        "K":K,
        "temperature":temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])
    prediction=model.predict(input_df)[0]
    st.success(f"Recommended: {prediction}")
        
    st.session_state["history"].append({
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "crop": prediction
    })


if st.session_state["history"]:
    st.info("📜 Prediction History:")
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df)

if st.button("Clear History"):
    st.session_state["history"] = []
    st.info("History cleared ✅")
    st.rerun()