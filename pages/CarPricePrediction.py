
import streamlit as st
import joblib
import pandas as pd
import numpy as np

import sys

# إضافة مسار سكريبت الـ FrequencyEncoder لو معمول له import داخل الـ pipeline
sys.path.append("H:/Final project") 
from frequency_encoder import FrequencyEncoder  # تأكد إن الملف اسمه كده فعلاً

# تحميل الموديل والبيانات المطلوبة
pipeline = joblib.load("H:/Final project/car_price_stacking_ML_model.h5")
BRANDLIST = joblib.load('H:/Final project/H5/BrandList.h5')
BODYLIST = joblib.load('H:/Final project/H5/BodyList.h5')
STATElist = joblib.load('H:/Final project/H5/StateList.h5')
COLORLIST = joblib.load("H:/Final project/H5/ColorList.h5")
INTERIORLIST = joblib.load("H:/Final project/H5/InteriorList.h5")
MODELLIST = joblib.load("H:/Final project/H5/ModelList.h5")
TIMEPERIOD = joblib.load("H:/Final project/H5/TimeperiodList.h5")

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
    brand = st.selectbox('Brand', sorted(BRANDLIST))
    model = st.selectbox('Model', sorted(MODELLIST))
    body = st.selectbox('Body Type', sorted(BODYLIST))
    transmission = st.selectbox('Transmission', ['automatic', 'manual'])
    state = st.selectbox('State', sorted(STATElist))
    condition = st.slider('Condition', min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    motor_mi = st.slider('Motor Mileage (mi)', min_value=0.0, max_value=205000.0, value=90000.0)
    color = st.selectbox('Exterior Color', sorted(COLORLIST))
    interior = st.selectbox('Interior Color', sorted(INTERIORLIST))

    # تاريخ البيع
    sell_year = st.number_input('Sell Year', min_value=2014, max_value=2016, value=2015)
    sell_month = st.number_input('Sell Month', min_value=1, max_value=12, value=2)
    sell_day = st.number_input('Sell Day', min_value=1, max_value=31, value=15)
    time_period = st.selectbox('Time Period', TIMEPERIOD)

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
