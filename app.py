import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

@st.cache_data
def load_data():
    with open("influenzaUkraine.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    records = []
    for year_entry in data["years"]:
        year = year_entry["year"]
        for item in year_entry["data"]:
            week = item["week"]
            if 1 <= week <= 52:
                cases = item["cases"]
                date = pd.to_datetime(f"{year}-W{int(week)}-1", format="%G-W%V-%u")
                records.append({"ds": date, "y": cases})
    df = pd.DataFrame(records).sort_values("ds")
    return df

@st.cache_data
def load_kharkiv_data():
    with open("influenzaKharkiv.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["data"])
    df["ds"] = pd.to_datetime(df["date"])

    df = df[["ds", "value"]]
    df.set_index("ds", inplace=True)

    df = df.resample("W").mean()
    df["y"] = df["value"].interpolate(method="linear")
    df.reset_index(inplace=True)

    return df[["ds", "y"]]

def forecast_sarima(df, weeks):
    df_sarima = df.copy()
    df_sarima.set_index('ds', inplace=True)

    model = SARIMAX(df_sarima['y'], order=(0, 1, 0), seasonal_order=(2, 1, 0, 52))
    model_fit = model.fit(disp=False)

    forecast = model_fit.get_forecast(steps=weeks)
    forecast_values = forecast.predicted_mean.clip(lower=0)
    forecast_index = pd.date_range(start=df['ds'].max() + pd.Timedelta(weeks=1), periods=weeks, freq='W-MON')
    return pd.DataFrame({'ds': forecast_index, 'y': forecast_values.values})

def forecast_prophet(df, weeks):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=weeks, freq='W')
    forecast = model.predict(future)
    return forecast

st.title("Прогнозування захворюваності")

model_type = st.selectbox("Оберіть модель прогнозування", ['SARIMA', 'Prophet'])
region = st.selectbox("Оберіть регіон", ['Україна', 'Харківська область'])
weeks = st.slider("На скільки тижнів прогнозувати?", min_value=1, max_value=52, value=12)

if region == "Україна":
    df = load_data()
else:
    df = load_kharkiv_data()

if st.button("Виконати прогноз"):
    if model_type == "Prophet":
        forecast = forecast_prophet(df, weeks)
        forecast_tail = forecast.tail(weeks)
        st.subheader(f"Прогноз Prophet ({region})")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["ds"], df["y"], "k-", label="Фактичні дані")
        ax.plot(forecast_tail["ds"], forecast_tail["yhat"], "b-", label="Прогноз")
        ax.fill_between(forecast_tail["ds"], forecast_tail["yhat_lower"], forecast_tail["yhat_upper"],
                        color="blue", alpha=0.2, label="Довірчий інтервал (80%)")
        ax.set_title(f"Прогноз випадків захворюваності Prophet ({region})")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Кількість випадків")
        ax.legend()
        st.pyplot(fig)

    elif model_type == "SARIMA":
        forecast = forecast_sarima(df, weeks)
        st.subheader(f"Прогноз SARIMA ({region})")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["ds"], df["y"], "k-", label="Фактичні дані")
        ax.plot(forecast["ds"], forecast["y"], "r-", label="Прогноз")
        ax.set_title(f"Прогноз випадків захворюваності SARIMA ({region})")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Кількість випадків")
        ax.legend()
        st.pyplot(fig)
