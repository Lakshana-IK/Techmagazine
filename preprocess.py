import pandas as pd

def load_and_clean_data(csv_file):
    """
    Loads and preprocesses the agricultural market data.
    """
    try:
        df = pd.read_csv(csv_file)

        # Standard column names (optional)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Drop rows with missing essential values
        df.dropna(subset=['commodity', 'price', 'date', 'region'], inplace=True)

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows where date conversion failed
        df.dropna(subset=['date'], inplace=True)

        # Extract features like year and month for modeling
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Ensure price is numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(subset=['price'], inplace=True)

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()