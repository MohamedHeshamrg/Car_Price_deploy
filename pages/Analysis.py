
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


st.set_page_config(page_title="Explatory Data Analysis", page_icon="📈", layout="wide")

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
            📈 Explatory Data Analysis 📊
        </div>
    """, unsafe_allow_html=True)

heading()

st.markdown("""
    <style>
    [data-testid=metric-container] {
        box-shadow: 0 0 4px #cccccc;
        padding: 10px;
    }

    .plot-container > div {
        box-shadow: 0 0 4px #cccccc;
        padding: 10px;
    }

    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.3rem;
        color: rgb(71, 146, 161);
    }
    </style>
""", unsafe_allow_html=True)



st.markdown("##")

tab1, tab2 ,tab3 = st.tabs(['📊🟦 Categorical Analysis','📈🟧 Numerical Analysi', '📈🟩 Bi/multi-variate Analysis'])



# Reading parts
@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/MohamedHeshamrg/Car_Price/main/Data/Data%20cleaned/"
    
    df1 = pd.read_csv(base_url + "part1.csv")
    df2 = pd.read_csv(base_url + "part2.csv")
    df3 = pd.read_csv(base_url + "part3.csv")
    df4 = pd.read_csv(base_url + "part4.csv")
    
    return pd.concat([df1, df2, df3, df4], ignore_index=True)

df = load_data()



# ==============================
# 📊 Univariate Analysis
# ==============================
st.sidebar.header("📊 Analysis")



# ------------------------------
# 🟦 Categorical
# ------------------------------
with tab1:
    st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Categorical features</h3>', unsafe_allow_html=True)
    sts = st.selectbox('select How featureImpact on :',
                       ['Number of sales', 'Revenue'], key=21)
    if sts == 'Number of sales':
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Vehicles identity features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select Vehicles identity feature 🚗 to see its distribution : ',
                      ['brand', 'model', 'trim', 'body'], key=20)
            col1, col2 = st.columns([4,3])
    
            data = df[col].value_counts()
            data = pd.DataFrame(data)
            data.reset_index(inplace=True)
            fig = px.bar(data[:15],y=col, x = 'count'
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 15 Most Frequent Values in {col}"
                )
        
            col1.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values='count',
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col2.plotly_chart(fig, use_container_width=True)
            
            
            
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Vehicle Attributes features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Vehicle Attributes feature 👀 to see its distribution : ',
                      ['model_year','color', 'interior','transmission'], key=23)
            col3, col4 = st.columns([4,3])
    
            data = df[col].value_counts()
            data = pd.DataFrame(data)
            data.reset_index(inplace=True)
            fig = px.bar(data[:20],x=col, y = 'count'
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 15 Most Frequent Values in {col}"
                )
        
            col3.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values='count',
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col4.plotly_chart(fig, use_container_width=True)

        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Location features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Sell Location feature 🚩 to see its distribution : ',
                      ['state','seller'], key=25)
            col5, col6 = st.columns([4,3])
    
            data = df[col].value_counts()
            data = pd.DataFrame(data)
            data.reset_index(inplace=True)
            fig = px.bar(data[:20],x=col, y = 'count'
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 15 Most Frequent Values in {col}"
                )
        
            col5.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values='count',
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col6.plotly_chart(fig, use_container_width=True)
            
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Sales time features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Sell Sales time feature ⌚ to see its distribution : ',
                      ['sell_day_name','sell_month_name','sell_year','season', 'time_period'], key=26)
            col7, col8 = st.columns([4,3])
    
            data = df[col].value_counts()
            data = pd.DataFrame(data)
            data.reset_index(inplace=True)
            fig = px.bar(data,x=col, y = 'count'
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Most Frequent Values in {col}"
                )
        
            col7.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values='count',
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col8.plotly_chart(fig, use_container_width=True)





    else:
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Vehicles identity features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select Vehicles identity 🚗 feature to see its distribution : ',
                      ['brand', 'model', 'trim', 'body'], key=22)
            col1, col2 = st.columns([4,3])
    
            data = df.groupby(col)['sellingprice'].sum().sort_values(ascending=False).reset_index()
 
            fig = px.bar(data[:15],y=col, x = "sellingprice"
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 15 Categories by Revenue in {col}"
                )
        
            col1.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values= "sellingprice",
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col2.plotly_chart(fig, use_container_width=True)

        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Vehicle Attributes features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select Vehicle Attributes 👀 feature to see its distribution : ',
                      ['model_year','color', 'interior','transmission'], key=24)
            col3, col4 = st.columns([4,3])
    
            data = df.groupby(col)['sellingprice'].sum().sort_values(ascending=False).reset_index()
 
            fig = px.bar(data[:20],x=col, y = "sellingprice"
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 20 Categories by Revenue in {col}"
                )
        
            col3.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values= "sellingprice",
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col4.plotly_chart(fig, use_container_width=True)

        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Location features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select Sell Location 🚩 feature to see its distribution : ',
                      ['state','seller'], key=25)
            col5, col6 = st.columns([4,3])
    
            data = df.groupby(col)['sellingprice'].sum().sort_values(ascending=False).reset_index()
 
            fig = px.bar(data[:20],x=col, y = "sellingprice"
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 20 Categories by Revenue in {col}"
                )
        
            col5.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values= "sellingprice",
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col6.plotly_chart(fig, use_container_width=True)

        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Sales time features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Sell Sales time feature ⌚ to see its distribution : ',
                      ['sell_day_name','sell_month_name','sell_year','season', 'time_period'], key=26)
            col7, col8 = st.columns([4,3])
    
            data = df.groupby(col)['sellingprice'].sum().sort_values(ascending=False).reset_index()
 
            fig = px.bar(data,x=col, y = "sellingprice"
                ,color_discrete_sequence=['#5ab8db'],
                title=f"Top 20 Categories by Revenue in {col}"
                )
        
            col7.plotly_chart(fig, use_container_width= True)

        
        
            fig = px.pie(data[:10], names=col,values= "sellingprice",
                      color_discrete_sequence=px.colors.sequential.PuBu,
                      title = f"Share of Top 10 Each Category in {col}"
                )
            col8.plotly_chart(fig, use_container_width=True)

        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of Performace </h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select feature  to see its distribution : ',
                      ['sell_day_name','sell_month_name','sell_year','season', 'time_period','state','seller',
                       'model_year','color', 'interior','transmission','brand', 'model', 'trim', 'body'], key=27)
      
        
    
            # How Brand performace
            top_makes = df[col].value_counts().nlargest(10).index

            filtered_data = df[df[col].isin(top_makes)]

            fig = px.box(filtered_data, x=col, y='sellingprice',
                title=f"Selling Price Distribution by {col}",
             category_orders={col: top_makes.tolist()})
            fig.update_layout(yaxis_title="Selling Price", xaxis_title= col, width=1100, height=600)
        
            st.plotly_chart(fig, use_container_width= True)



                
    st.write("📌 **Statistics for Categorical Columns**")
    st.dataframe(df.describe(include="O").T)



# ------------------------------
# 🟧 Numerical
# ------------------------------
with tab2:
    st.markdown(
        '<h3 style="text-align: center; color : #5ab8db;">📊 Charts of Numerical Features</h3>',
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown(
            '<h4 style="text-align: center; color : #5ab8db;">📈 Distribution of Numerical Features</h4>',
            unsafe_allow_html=True
        )

        col = st.selectbox(
            '🚗 Select a Numerical Feature to See Its Distribution:',
            ['sellingprice', 'mmr', 'motor_mi', 'market_advantage', 'condition'],
            key=30
        )


        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            histnorm='density',
            template="plotly_white",
            color_discrete_sequence=['#5ab8db']
        )

        fig.update_layout(
            title=f'Distribution  {col}',
            xaxis_title=col,
            yaxis_title='Density'
        )
        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)


        fig = px.box(
            df,
            x=col,  
            template="plotly_white",
            color_discrete_sequence=['#5ab8db'],
            title=f'Box Plot of {col}'
            )

        fig.update_layout(
            xaxis_title=col,
            )

        fig.update_xaxes(tickangle=45)  

        st.plotly_chart(fig, use_container_width=True)


        fig = px.violin(
            df,
            x=col,  
            template="plotly_white",
            color_discrete_sequence=['#5ab8db'],
            title=f' Violin Plot of {col}'
            )

        fig.update_layout(
            xaxis_title=col,
            )

        fig.update_xaxes(tickangle=45)  

        st.plotly_chart(fig, use_container_width=True)
        

        
        

    st.write("📌 **Statistics For Numerical Feature**")
    st.dataframe(df.describe().T)


# ------------------------------
# 🟧  Bi/multi-variate Analysis
# ------------------------------
with tab3:
    st.markdown(
        '<h3 style="text-align: center; color : #5ab8db;">📊 Charts of Bi/multi-variate Analysis</h3>',
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown(
            '<h4 style="text-align: center; color : #5ab8db;">HeatMap Of Numerical Features</h4>',
            unsafe_allow_html=True
        )


        data = df.corr(numeric_only=True)
        fig= px.imshow(data,height=800 , width= 800, title= "HeatMap"
                   ,color_continuous_scale=px.colors.sequential.Blues)
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<h4 style="text-align: center; color : #5ab8db;"> Condition vs Market Advantage Features</h4>',
            unsafe_allow_html=True
        )

        df['below_mmr'] = df['market_advantage'] < 0
        df['below_mmr'] = df['below_mmr'].map({True: 'Below MMR', False: 'Above/Equal MMR'})

        fig = px.box(df, x='below_mmr', y='condition',
             color='below_mmr',
             title='Condition vs Market Advantage',
             color_discrete_sequence=['#5ab8db', '#023e8a'],
             template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        col = st.selectbox(
            '🚗 Select a Feature to See Its Average Selling Price :',
            ['model_year' , 'condition', 'sell_day','sell_month'],
            key=40
        )

        avg_prices = df.groupby(col)['sellingprice'].mean().reset_index()

        fig = px.line(
        avg_prices,
        x=col,
        y='sellingprice',
        markers=True,
        title=f'Average Selling Price by {col}',
        template='plotly_white',
        line_shape='spline',
        color_discrete_sequence=['#5ab8db']
            )

        fig.update_layout(xaxis_title=col, yaxis_title='Average Selling Price')
        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

        col = st.selectbox(
            '🚗 Select a Feature to See Its Average Selling Price with Transmmision :',
            ['model_year' , 'condition', 'sell_day','sell_month',"state"],
            key=41
        )
        # Model year and Value color by Transmission
        YT = pd.crosstab(df[col],df["transmission"])
        fig = px.line(YT,title=f"Distribution of {col} preferance color by Transmission all cars")
        st.plotly_chart(fig, use_container_width=True)

        col = st.selectbox(
            '🚗 Select a Feature to See Its Average Selling Price with Trim :',
            ["brand","model"],
            key=42)

        brand_trim_price = df.groupby([col, 'trim'])['sellingprice'].mean().sort_values(ascending=False).reset_index()

        top_20 = brand_trim_price.head(20)

        fig = px.bar(
        top_20,
        x=col,
        y='sellingprice',
        color='trim',
        title=f'Top 20 Expensive Trims by {col}',
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Blues
            )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title='Average Selling Price',
            legend_title='Trim',
            xaxis_tickangle=45
            )

        st.plotly_chart(fig, use_container_width=True)


        model_impact = df.groupby('model')[['market_advantage', 'sellingprice']].mean().reset_index()
        top_models = df['model'].value_counts().nlargest(30).index
        model_impact_top = model_impact[model_impact['model'].isin(top_models)]

        fig = px.scatter(model_impact_top,
                 x='market_advantage',
                 y='sellingprice',
                 text='model',
                 size='sellingprice',
                 color='model',
                 title='Market Advantage vs Selling Price by Model',
                 width=1000,
                 height=600)
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        model_impact = df.groupby('model')[['market_advantage', 'sellingprice']].mean().reset_index()

        top_models = model_impact[model_impact['market_advantage'] > 0].sort_values('market_advantage', ascending=False).head(20)


        fig = px.scatter(top_models,
                 x='market_advantage',
                 y='sellingprice',
                 text='model',
                 size='sellingprice',
                 color='model',
                 title='Top 30 Models with Positive Market Advantage',
                 width=1000,
                 height=600)

        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Condition Impact on Price within Top Brands
        top_makes = ['Nissan', 'Ford', 'Chevrolet', 'Toyota', 'BMW','Lexus']

        condition_price_impact = df[df['brand'].isin(top_makes)].groupby(['brand', 'condition'])['sellingprice'].mean().unstack()

        fig = px.line(condition_price_impact.T, title='Condition Impact on Price within Top Brands', width=1100, height=400)
        fig.update_layout(xaxis_title='Condition', yaxis_title='Average Selling Price', legend_title='Brand')
        st.plotly_chart(fig, use_container_width=True)


#1- How does the condition and model year together influence the selling price?

        condition_price_impact = df.groupby(['model_year', 'condition'])['sellingprice'].mean().reset_index()

        fig = px.line(condition_price_impact,
              x='condition',
              y='sellingprice',
              color='model_year',
              title='Condition and Model Year Impact on Selling Price',
              markers=True,height=800,width=1100)
        fig.update_layout(xaxis_title='Condition', yaxis_title='Average Selling Price', legend_title='Model Year')
        st.plotly_chart(fig, use_container_width=True)

