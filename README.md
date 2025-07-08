# 🌾 Agricultural Market Price Analyzer

This is a data-driven Streamlit dashboard for analyzing and predicting agricultural commodity prices. It supports farmers, traders, policymakers, and agri-tech startups by providing real-time insights, historical trends, and machine learning–based price forecasting.

---

## 🚀 Features
- Real-time commodity price monitoring
- Historical price trends with interactive charts
- Weather impact analysis (temperature, rainfall)
- Price forecasting using machine learning (XGBoost, scikit-learn)
- Regional market comparison via mapping
- Exportable PDF reports

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Folium
- **ML Models**: XGBoost, scikit-learn
- **Weather Integration**: OpenWeatherMap API (or similar)
- **Reporting**: FPDF

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/agri-market-analyzer.git
cd agri-market-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
