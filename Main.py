import streamlit as st
import pandas as pd 
import plotly.express as px
import time
from numerize.numerize import numerize
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import joblib
from scipy.stats import gaussian_kde
import sys

import joblib
import pandas as pd
import numpy as np

import sys
# إضافة مسار سكريبت الـ FrequencyEncoder لو معمول له import داخل الـ pipeline
sys.path.append("H:/Final project") 
from frequency_encoder import FrequencyEncoder 

sys.path.append("H:/Final project/Deployment")

from UI import *
#page behaviour

st.set_page_config(page_title="Descriptive Analytics ", page_icon="🌎", layout="wide")  

# Custom heading with dark blue background and white text
def heading():
    st.markdown("""  
        <style>
        .custom-heading {
            background-color: #5ab8db;  /* لبني غامق */
            color: white;              /* خط أبيض */
            padding: 20px;
            border-radius: 12px;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
        }
        </style>

        <div class="custom-heading">
            📈 Descriptive Analytics 📊
        </div>
    """, unsafe_allow_html=True)




#remove default theme
theme_plotly = None # None or streamlit

 
# CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# Reading parts
@st.cache_data
def load_data():
    df1 = pd.read_csv('H:/Final project/Data/Data cleaned/part1.csv')
    df2 = pd.read_csv('H:/Final project/Data/Data cleaned/part2.csv')
    df3 = pd.read_csv('H:/Final project/Data/Data cleaned/part3.csv')
    df4 = pd.read_csv('H:/Final project/Data/Data cleaned/part4.csv')
    df5 = pd.read_csv('H:/Final project/Data/Data cleaned/part5.csv')
    df6 = pd.read_csv('H:/Final project/Data/Data cleaned/part6.csv')

    
    return pd.concat([df1, df2, df3, df4,df5,df6], ignore_index=True)

df = load_data()






def HomePage():
 heading()
  #1. print dataframe
 with st.expander("🧭 My database"):
  #st.dataframe(df_selection,use_container_width=True)
  st.dataframe(df,use_container_width=True)

 #2. compute top Analytics
 
 total_Sales = 558
 Total_Ravenu = 760
 Best_selling_brand = 103
 The_most_expensive_car_sold= 18.3	 

 #3. columns
 total1,total2,total3,total4 = st.columns(4,gap='large')
 with total1:

    st.info('Total Sales', icon="🔎")
    st.metric(label = 'Count', value= f"{total_Sales}K")
    
 with total2:
    st.info('Total Ravenu', icon="💵")
    st.metric(label='Sum', value=f"{Total_Ravenu}M")

 with total3:
    st.info('Best Selling Brand', icon="🚗")
    st.metric(label= 'Mean TZS',value=f"{Best_selling_brand}K")

 with total4:
    st.info('Most Expensive Car', icon="💸")
    st.metric(label='Ferrari',value=f"{The_most_expensive_car_sold}K")


    
 st.markdown("""---""")

 #graphs
 
def Graphs():
 

   with st.container():
      col1, col2 = st.columns([4,3])
      df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)

# Verify column type
      if isinstance(df['saledate'].dtype, pd.DatetimeTZDtype):
         df['saledate'] = df['saledate'].dt.tz_localize(None)
         df['sale_month'] = df['saledate'].dt.to_period('M').dt.to_timestamp()

      monthly_avg = df.groupby('sale_month')['sellingprice'].mean().reset_index()

      fig = px.line(
         monthly_avg,
          x='sale_month',
         y='sellingprice',
         title='📈 Average Selling Price per Month',
         markers=True,
         template='plotly_white',
         color_discrete_sequence=['#5ab8db']
            )

      fig.update_layout(
         xaxis_title='Sale Month',
         yaxis_title='Average Selling Price',
            )
      fig.update_xaxes(tickangle=45)

      col1.plotly_chart(fig, use_container_width=True)

      fig = px.histogram(
            df,
            x='sellingprice',
            nbins=30,
            histnorm='density',
            template="plotly_white",
            color_discrete_sequence=['#5ab8db']
        )

      fig.update_layout(
            title=f'Distribution Selling price',
            xaxis_title="Selling Price",
            yaxis_title='Density'
        )
      fig.update_xaxes(tickangle=45)

      col2.plotly_chart(fig, use_container_width=True)



   with st.container():
      col1, col2 = st.columns([4,3])
        # Condition Impact on Price within Top Brands
      top_makes = ['Nissan', 'Ford', 'Chevrolet', 'Toyota', 'BMW','Lexus']

      condition_price_impact = df[df['brand'].isin(top_makes)].groupby(['brand', 'condition'])['sellingprice'].mean().unstack()

      fig = px.line(condition_price_impact.T, title='Condition Impact on Price within Top Brands', width=1100, height=400)
      fig.update_layout(xaxis_title='Condition', yaxis_title='Average Selling Price', legend_title='Brand')
      col1.plotly_chart(fig, use_container_width=True)



      data = df.groupby("brand")['sellingprice'].sum().sort_values(ascending=False).reset_index()
      fig = px.pie(data[:10], names="brand",values= "sellingprice",
                      color_discrete_sequence=px.colors.sequential.PuBu, hole=0.4,
                      title = "Share of Top 10 Each Category in Brand"
                )
      col2.plotly_chart(fig, use_container_width=True)

   # Statewise Preferance color by Transmission 
   ST = pd.crosstab(df["state"],df["transmission"])
   fig = px.line(ST, title= "Statewise Transmission preferance")
   st.plotly_chart(fig, use_container_width=True)


HomePage()
Graphs()
     
  

footer="""<style>
 

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
height:5%;
bottom: 0;
width: 100%;
background-color: #243946;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with  ❤ by Eng. Mohamed Hesham Ragab<a style='display: block; text-align: center;target="_blank"></a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
