
import streamlit as st
import joblib
import pandas as pd
import numpy as np

import sys

import sys
import requests
import tempfile
import joblib
import importlib.util

# تحميل ملف frequency_encoder.py من GitHub
url_encoder = "https://raw.githubusercontent.comMohamedHeshamrg/Car_Price/main/frequency_encoder.py"
response = requests.get(url_encoder)
response.raise_for_status()

# حفظ الملف مؤقتًا
with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_encoder:
    tmp_encoder.write(response.content)
    encoder_path = tmp_encoder.name

# استيراد FrequencyEncoder ديناميكياً
spec = importlib.util.spec_from_file_location("frequency_encoder", encoder_path)
freq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freq_module)
FrequencyEncoder = freq_module.FrequencyEncoder

# تحميل ملف الموديل h5 من GitHub
url_model = "https://raw.githubusercontent.com/MohamedHeshamrg/Car_Price/main/car_price_stacking_ML_model.h5"
response = requests.get(url_model)
response.raise_for_status()

# حفظ ملف h5 مؤقتًا وتحميله
with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_model:
    tmp_model.write(response.content)
    model_path = tmp_model.name

pipeline = joblib.load(model_path)


# الأعمدة المطلوبة في الـ input
inputs = ['brand', 'model', 'model_year', 'body', 'transmission', 'state',
          'condition', 'motor_mi', 'color', 'interior',
          'sell_year', 'sell_month', 'sell_day', 'time_period']

# دالة التنبؤ
def predict(model_year, brand, model, body, transmission, state,
            condition, motor_mi, color, interior,
            sell_year, sell_month, sell_day, time_period):

    # بناء DataFrame واحد للمدخلات
    test_df = pd.DataFrame([{
        'brand': brand,
        'model': model,
        'model_year': model_year,
        'body': body,
        'transmission': transmission,
        'state': state,
        'condition': condition,
        'motor_mi': np.log1p(motor_mi),
        'color': color,
        'interior': interior,
        'sell_year': sell_year,
        'sell_month': sell_month,
        'sell_day': sell_day,
        'time_period': time_period
    }])

    # التنبؤ بالسعر مباشرة من الـ pipeline
    log_predicted_price = pipeline.predict(test_df)[0]
    predicted_price = np.expm1(log_predicted_price)

    return predicted_price

# واجهة Streamlit
def main():
    st.title('🚗 Car Price Prediction Based on Car Specifications')

    # مدخلات المستخدم
    model_year = st.number_input('Model Year', min_value=1982, max_value=2016, value=2008)
    brand = st.selectbox('Brand', sorted(df['brand'].unique()))
    model = st.selectbox('Model', sorted(df['model'].unique()))
    body = st.selectbox('Body Type', sorted(df['body'].unique()))
    transmission = st.selectbox('Transmission', ['automatic', 'manual'])
    state = st.selectbox('State', sorted(df['state'].unique()))
    condition = st.slider('Condition', min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    motor_mi = st.slider('Motor Mileage (mi)', min_value=0.0, max_value=205000.0, value=90000.0)
    color = st.selectbox('Exterior Color', sorted(df['color'].unique()))
    interior = st.selectbox('Interior Color', sorted(df['interior'].unique()))

    # تاريخ البيع
    sell_year = st.number_input('Sell Year', min_value=2014, max_value=2016, value=2015)
    sell_month = st.number_input('Sell Month', min_value=1, max_value=12, value=2)
    sell_day = st.number_input('Sell Day', min_value=1, max_value=31, value=15)
    time_period = st.selectbox('Time Period', df['time_period'].unique())

    # زر التنبؤ
    if st.button('🔮 Predict Price'):
        predicted_price = predict(
            model_year, brand, model, body, transmission, state,
            condition, motor_mi, color, interior,
            sell_year, sell_month, sell_day, time_period
        )
        st.success(f"✅ Estimated Car Price: ${predicted_price:,.2f}")

if __name__ == '__main__':
    main()
