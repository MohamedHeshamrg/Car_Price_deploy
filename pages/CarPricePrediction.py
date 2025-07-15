
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  RobustScaler
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge , Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder


from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_maps = {
            col: X[col].value_counts(normalize=True)
            for col in X.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X

@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/MohamedHeshamrg/Car_Price/main/data/"
    
    df1 = pd.read_csv(base_url + "part1.csv")
    df2 = pd.read_csv(base_url + "part2.csv")
    df3 = pd.read_csv(base_url + "part3.csv")
    df4 = pd.read_csv(base_url + "part4.csv")
    df5 = pd.read_csv(base_url + "part5.csv")
    df6 = pd.read_csv(base_url + "part6.csv")
    df = pd.concat([df1, df2, df3, df4,df5,df6], ignore_index=True)
    return df

df = load_data()

df['motor_mi'] = np.log1p(df['motor_mi'])
df.drop(['seller','saledate','market_advantage','sell_month_name','sell_day_name','sell_hour','trim', 'season','mmr'],axis = 1 , inplace = True)
# Condition Values change from 0 to 50 --> 1 to 5 
df['condition'] = pd.cut(df['condition'], bins=[0, 10, 20, 30, 40, 50], labels=[1, 2, 3, 4, 5])
df['condition'] = df['condition'].astype(int)

x = df.drop('sellingprice', axis = 1)
y = df['sellingprice']

# StackingRegressor Final Model

# ------------ Column Division ------------
freq_cols = ['brand', 'model', 'body', 'state', 'color', 'interior', 'time_period']
binary_cols = ['transmission']
scale_cols = ['motor_mi', 'condition', 'model_year','sell_year', 'sell_month', 'sell_day']
from sklearn.linear_model import Ridge, Lasso



# ------------ ColumnTransformer ------------
preprocessor = ColumnTransformer(transformers=[
    ('freq_enc', Pipeline([
        ('freq', FrequencyEncoder()),
        ('scale', RobustScaler())
    ]), freq_cols),
    ('binary_enc', BinaryEncoder(), binary_cols),
    ('num_scaler', RobustScaler(), scale_cols)
], remainder='passthrough')

# ------------ Base Models ------------
xgb = XGBRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        reg_alpha=0.5,
        reg_lambda=1.0

)

lgbm = LGBMRegressor(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,

)



# ------------ Stacking Regressor ------------

ridge_base = Ridge(alpha=0.5)
lasso_base = Lasso(alpha=0.01)

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('ridge', ridge_base),
        ('lasso', lasso_base)
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1,
    passthrough=False
)
# ------------ Final Pipeline ------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', stacking_model)
])



scores = cross_validate(
    pipeline, x, np.log1p(y),  #  log target
    cv=5,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
    return_train_score=True
)
pipeline.fit(x, np.log1p(y))


# ------------ Display Results ------------

#Train R2: 0.9407939891644069
#Test R2: 0.919289751986786
#-------------------------
#Train MSE: 0.044501419409624786
#Test MSE: 0.06037161924907719
#-------------------------
#Train MAE: 0.14507140552622427
#Test MAE: 0.16628192948228704
#==================================================
#Run Time: 376.3431479999999




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
