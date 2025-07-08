import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

MODEL_PATH = "model.joblib"

def train_model(df):
    # Convert price to per kg
    if "weight" in df.columns:
        df["price_per_kg"] = df["price"] / df["weight"]
    else:
        df["price_per_kg"] = df["price"] / 100  # Assuming price is per quintal

    df = df.dropna(subset=["price_per_kg"])

    # Fill missing weather data with median
    df["avg_temp"].fillna(df["avg_temp"].median(), inplace=True)
    df["avg_humidity"].fillna(df["avg_humidity"].median(), inplace=True)

    if df.empty:
        raise ValueError("Training data is empty after preprocessing. Please check your dataset.")

    # Features and target
    X = df[["year", "month", "avg_temp", "avg_humidity"]].copy()
    categorical_cols = ["commodity", "region"]

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)
    y = df["price_per_kg"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoder
    joblib.dump((model, encoder), MODEL_PATH)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def load_model():
    model, encoder = joblib.load(MODEL_PATH)
    return model, encoder

def predict_price(model_and_encoder, input_dict):
    model, encoder = model_and_encoder
    df_input = pd.DataFrame([input_dict])

    # Extract and encode categorical features
    categorical_df = pd.DataFrame([[input_dict["commodity"], input_dict["region"]]], columns=["commodity", "region"])
    encoded = encoder.transform(categorical_df)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["commodity", "region"]))

    # Combine all features
    X_input = pd.DataFrame([[
        input_dict["year"],
        input_dict["month"],
        input_dict["avg_temp"],
        input_dict["avg_humidity"]
    ]], columns=["year", "month", "avg_temp", "avg_humidity"])

    X_input = pd.concat([X_input, encoded_df], axis=1)

    # Predict
    prediction = model.predict(X_input)[0]
    return prediction
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

MODEL_PATH = "model.joblib"

def train_model(df):
    # Convert price to per kg
    if "weight" in df.columns:
        df["price_per_kg"] = df["price"] / df["weight"]
    else:
        df["price_per_kg"] = df["price"] / 100  # Assuming price is per quintal

    df = df.dropna(subset=["price_per_kg"])

    # Fill missing weather data with median
    df["avg_temp"].fillna(df["avg_temp"].median(), inplace=True)
    df["avg_humidity"].fillna(df["avg_humidity"].median(), inplace=True)

    if df.empty:
        raise ValueError("Training data is empty after preprocessing. Please check your dataset.")

    # Features and target
    X = df[["year", "month", "avg_temp", "avg_humidity"]].copy()
    categorical_cols = ["commodity", "region"]

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)
    y = df["price_per_kg"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoder
    joblib.dump((model, encoder), MODEL_PATH)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def load_model():
    model, encoder = joblib.load(MODEL_PATH)
    return model, encoder

def predict_price(model_and_encoder, input_dict):
    model, encoder = model_and_encoder
    df_input = pd.DataFrame([input_dict])

    # Extract and encode categorical features
    categorical_df = pd.DataFrame([[input_dict["commodity"], input_dict["region"]]], columns=["commodity", "region"])
    encoded = encoder.transform(categorical_df)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["commodity", "region"]))

    # Combine all features
    X_input = pd.DataFrame([[
        input_dict["year"],
        input_dict["month"],
        input_dict["avg_temp"],
        input_dict["avg_humidity"]
    ]], columns=["year", "month", "avg_temp", "avg_humidity"])

    X_input = pd.concat([X_input, encoded_df], axis=1)

    # Predict
    prediction = model.predict(X_input)[0]
    return prediction