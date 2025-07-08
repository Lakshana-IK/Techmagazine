import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os
from PIL import Image, ImageDraw
from datetime import datetime
from utils.preprocess import load_and_clean_data
from utils.prediction import train_model, load_model, predict_price
from utils.weather import add_weather_to_df
from utils.report import create_report

os.makedirs("images", exist_ok=True)

def create_placeholder_image(path, text="Chart not available"):
    img = Image.new("RGB", (640, 480), color="gray")
    draw = ImageDraw.Draw(img)
    draw.text((100, 220), text, fill="black")
    img.save(path)

st.set_page_config(page_title="Agri Market Price Analyzer", layout="wide")
st.title("🌾 Agricultural Market Price Analyzer")

st.sidebar.header("📁 Upload Market Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

region_coords = {
    "Punjab": (31.1471, 75.3412),
    "Maharashtra": (19.7515, 75.7139),
    "Tamil Nadu": (11.1271, 78.6569),
    "Karnataka": (15.3173, 75.7139),
    "Gujarat": (22.2587, 71.1924),
}

if uploaded_file:
    df = load_and_clean_data(uploaded_file)

    if not df.empty:
        st.success("✅ Data loaded successfully!")

        with st.spinner("Adding weather data..."):
            df = add_weather_to_df(df, region_coords)

        df = df.infer_objects(copy=False)
        df["avg_temp"] = df["avg_temp"].fillna(30.0)
        df["avg_humidity"] = df["avg_humidity"].fillna(70.0)

        # Then explicitly convert types if needed
        df = df.infer_objects(copy=False)


        # Convert price to per kg if needed
        if "weight" in df.columns:
            df["price_per_kg"] = df["price"] / df["weight"]
        else:
            df["price_per_kg"] = df["price"] / 100  # Assuming price is per quintal

        st.dataframe(df.head())

        st.sidebar.header("🔍 Filter Options")
        selected_commodity = st.sidebar.selectbox("Select Commodity", df['commodity'].unique())
        selected_region = st.sidebar.selectbox("Select Region", df['region'].unique())

        filtered_df = df[
            (df["commodity"] == selected_commodity) &
            (df["region"] == selected_region)
        ]

        st.subheader("📈 Price Trend")
        fig_price = px.line(filtered_df, x="date", y="price_per_kg", title="Price per kg Over Time", markers=True)
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("🌡️ Temperature vs Price")
        fig_temp = px.scatter(filtered_df, x="avg_temp", y="price_per_kg", trendline="ols", title="Temp vs Price per kg")
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("💧 Humidity vs Price")
        fig_humidity = px.scatter(filtered_df, x="avg_humidity", y="price_per_kg", trendline="ols", title="Humidity vs Price per kg")
        st.plotly_chart(fig_humidity, use_container_width=True)

        st.subheader("🤖 Train Price Prediction Model")
        if st.button("🚀 Train Model"):
            try:
                mse = train_model(df)
                st.success(f"Model trained! MSE: {mse:.2f}")
            except ValueError as e:
                st.error(str(e))

        st.markdown("### 🔮 Predict Future Price")
        year = st.number_input("Year", min_value=2024, max_value=2030, value=2025)
        month = st.selectbox("Month", list(range(1, 13)))
        commodity = st.selectbox("Commodity", df["commodity"].unique())
        region = st.selectbox("Region", df["region"].unique())
        avg_temp = st.number_input("Avg Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        avg_humidity = st.number_input("Avg Humidity (%)", min_value=0, max_value=100, value=70)

        input_dict = {
            "year": year,
            "month": month,
            "commodity": commodity,
            "region": region,
            "avg_temp": avg_temp,
            "avg_humidity": avg_humidity
        }

        if st.button("Predict Price"):
            try:
                model = load_model()
                result = predict_price(model, input_dict)
                st.session_state.prediction_result = result
                st.session_state.input_dict = input_dict

                st.success(f"📉 Predicted Price per kg: Rs.{result:.2f}")

                st.write("🔎 Input Used for Prediction:", input_dict)

                # 🔥 Save to Firebase with safety
                try:
                    save_prediction_to_firebase(input_dict, result)
                except Exception as firebase_error:
                    st.warning(f"Prediction saved locally but failed to sync with Firebase: {firebase_error}")

            except Exception as e:
                st.warning("⚠️ Prediction failed. Make sure the model is trained and the data includes 'price_per_kg'.")
                st.exception(e)

        st.subheader("📄 Export DOCX Report (Details Only)")
        if st.button("📥 Download Report"):
            try:
                with st.spinner("Generating report..."):
                    summary_text = (
                        f"Commodity: {selected_commodity}\n"
                        f"Region: {selected_region}\n"
                        f"Avg Price per kg: Rs.{filtered_df['price_per_kg'].mean():.2f}\n"
                        f"Avg Temp: {filtered_df['avg_temp'].mean():.2f} °C\n"
                        f"Avg Humidity: {filtered_df['avg_humidity'].mean():.2f} %\n"
                    )

                    if 'input_dict' in st.session_state:
                        input_dict = st.session_state.input_dict
                        summary_text += (
                            "\nPrediction Input:\n"
                            f"- Year: {input_dict['year']}\n"
                            f"- Month: {input_dict['month']}\n"
                            f"- Commodity: {input_dict['commodity']}\n"
                            f"- Region: {input_dict['region']}\n"
                            f"- Avg Temp: {input_dict['avg_temp']} °C\n"
                            f"- Avg Humidity: {input_dict['avg_humidity']} %\n"
                        )

                    if 'prediction_result' in st.session_state:
                        summary_text += f"\nPredicted Price per kg: Rs.{st.session_state.prediction_result:.2f}\n"

                    summary_text += f"\nReport Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

                    output_path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
                    create_report(summary_text, chart_paths=[], output_path=output_path)

                    with open(output_path, "rb") as f:
                        st.download_button("📄 Download Word Report", f, file_name="agri_price_report.docx")
            except Exception as e:
                st.error(f"Failed to export DOCX: {e}")

        st.subheader("🗂️ View Past Predictions")
        if st.button("📚 Show History"):
            history = get_all_predictions()
            if history:
                st.dataframe(pd.DataFrame(history))
            else:
                st.info("No prediction history found.")
    else:
        st.error("❌ Invalid or empty dataset.")
else:
    st.info("👈 Please upload a market CSV file to begin.")