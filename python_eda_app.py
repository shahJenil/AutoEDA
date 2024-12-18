import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Function to create correlation chart between variables in a df
def create_correlation_chart(corr_df):
    fig = plt.figure(figsize=(15,15))
    plt.imshow(corr_df.values, cmap='Blues')
    plt.xticks(range(corr_df.shape[0]),corr_df.columns,rotation=90,fontsize=15)
    plt.yticks(range(corr_df.shape[0]),corr_df.columns,fontsize=15)
    plt.colorbar()
    
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[0]):
            plt.text(i,j,"{:.2f}".format(corr_df.values[i,j]),color='red',ha='center',fontsize=14,fontweight='bold')
    
    return fig

# Function to create bar graphs for missing values
def missing_value_bar_chart(df):
    missing_fig = plt.figure(figsize=(10,5))
    ax = missing_fig.add_subplot(111)
    missingno.bar(df,figsize=(10,5),fontsize=15,ax=ax)
    
    return missing_fig

# Function to create histogram using Plotly
def create_histogram(df, feature):
    fig = px.histogram(
        df,
        x=feature,
        title=f'Distribution of {feature}',
        nbins=50,
        opacity=0.7
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title=feature,
        yaxis_title='Count',
        bargap=0.05
    )
    return fig

# Function to create bar chart using Plotly
def create_bar_chart(df_cnts):
    fig = px.bar(
        df_cnts,
        x='Type',
        y='Values',
        title='Category Distribution',
        color_discrete_sequence=['tomato']
    )
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Count',
        showlegend=False
    )
    fig.update_xaxes(tickangle=45)
    return fig

# Function to create scatter plot using Plotly
def create_scatter_plot(df, x_axis, y_axis, color_encode=None):
    if color_encode:
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_encode,
            title=f"{x_axis.capitalize()} vs {y_axis.capitalize()}",
            height=600
        )
    else:
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{x_axis.capitalize()} vs {y_axis.capitalize()}",
            height=600
        )
    
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        showlegend=True if color_encode else False
    )
    return fig

# Logic to separate categorical and continuous columns
def cat_and_cont_columns(df):
    cat_colums, cont_columns = [],[] 
    for col in df.columns:
        if len(df[col].unique()) <= 25 or df[col].dtype == np.object_:
            cat_colums.append(col.strip())
        else:
            cont_columns.append(col.strip())
    
    return cont_columns, cat_colums

# Initialising web app 
st.set_page_config(page_icon=":bar_chart:", page_title="Automated EDA using python", layout="wide")

upload = st.file_uploader(label="Upload file here : ", type=['csv'])

if upload:
    df = pd.read_csv(upload)
    cont_cols, cat_cols = cat_and_cont_columns(df)
    
    tab1, tab2, tab3 = st.tabs(["Dataset overview :clipboard:", "Individual column stats :bar_chart:", "Explore relation between features :chart:"])
    
    with tab1:
        st.subheader("1. Dataset")
        st.write(df)

        st.subheader("2. Dataset Overview")
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Rows",df.shape[0]),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Duplicates",df.shape[0]-df.drop_duplicates().shape[0]),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Features",df.shape[1]),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Categorical Columns",len(cat_cols)),unsafe_allow_html=True)
        st.write(cat_cols)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Continuous Columns",len(cont_cols)),unsafe_allow_html=True)
        st.write(cont_cols)

        st.subheader("3. Correlation chart")
        corr_df = df[cont_cols].corr()
        corr_fig = create_correlation_chart(corr_df)
        st.pyplot(corr_fig, use_container_width=True)

        st.subheader("4. Missing Values Distribution")
        missing_fig = missing_value_bar_chart(df)
        st.pyplot(missing_fig, use_container_width=True)

    # Individual column stats
    with tab2:
        df_desc = df.describe()
        st.subheader("Analyze Individual Feature Distribution")
        
        st.markdown("### 1. Analyze Continuous Features")
        feature = st.selectbox(label="Select Continuous Feature", options=cont_cols, index=0)
        
        na_cnt = df[feature].isna().sum()
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Count",df_desc[feature]['count']),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Missing Count",na_cnt),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Mean",df_desc[feature]['mean']),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Standard Deviation",df_desc[feature]['std']),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Maximum",df_desc[feature]['max']),unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Minimum",df_desc[feature]['min']),unsafe_allow_html=True)
        
        # Create histogram using Plotly
        hist_fig = create_histogram(df, feature)
        st.plotly_chart(hist_fig, use_container_width=True)
        
        st.markdown("### 2. Analyze Categorical Features")
        feature = st.selectbox(label="Select Categorical Feature", options=cat_cols, index=0)

        cnts = Counter(df[feature].dropna().values)
        df_cnts = pd.DataFrame({"Type": list(cnts.keys()), "Values": list(cnts.values())})
        
        # Create bar chart using Plotly
        bar_fig = create_bar_chart(df_cnts)
        st.plotly_chart(bar_fig, use_container_width=True)

    with tab3:
        st.subheader("Explore Relation Between Features of Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(label="X-Axis", options=cont_cols, index=0)
        with col2:
            y_axis = st.selectbox(label="Y-Axis", options=cont_cols, index=1)
            
        color_encode = st.selectbox(label='Color-Encode', options=[None] + cat_cols)
        
        # Create scatter plot using Plotly
        scatter_fig = create_scatter_plot(df, x_axis, y_axis, color_encode)
        st.plotly_chart(scatter_fig, use_container_width=True)