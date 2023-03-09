import streamlit as st
import pickle
import numpy as np


# Import Model

pipe= pickle.load(open("pipe.pkl", "rb"))
df= pickle.load(open("df.pkl", "rb"))

st.title("Laptop Price Predictor")

# Brand
company= st.selectbox("Brand", df["Company"].unique())

# Type
type= st.selectbox("TypeName", df["TypeName"].unique())

#Ram --> Manually providing the Ram
ram = st.selectbox("Ram ( in GB )", [2,4,6,8,12,16,24,32,64] )

#Weight
weight= st.number_input("Weight of the laptop")

#Touchscreen
touchscreen= st.selectbox("TouchScreen", ["No", "Yes"])

#Ips Display
Ips= st.selectbox("IPS Display", ["No", "Yes"])

#Screen size
screen_size= st.number_input("Screen Size")


#Screen Resolution
resolution= st.selectbox("Screen Resolution", ['1920x1080','1366x768','1600x900','3840x2160',
                                               '3200x1800','2880x1800','2560x1600',
                                               '2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['CPU Brand'].unique())

#HDD
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#SSD
ssd = st.selectbox("SSD(in GB)",[0,8,128,256,512,1024])

#GPU
gpu = st.selectbox('GPU',df["GPU Brand"].unique())

#Operating System
os = st.selectbox('Opearating System',df['OS'].unique())

if st.button('Predict Price'):
    # Query Point

    ppi= None

    # Ips and touchscreen is displayed and accepted as [ Yes or No ] but we need them as [ 1 or 0 ]

    if touchscreen == "Yes":
        touchscreen= 1
    else:
        touchscreen= 0

    if Ips == "Yes":
        Ips= 1
    else:
        Ips= 0

    # Resolution is String. Convert it to integer.
    X_Res= int( resolution.split("x")[0] )
    Y_Res= int( resolution.split("x")[1] )

    ppi= ( ( ( X_Res**2 ) + (Y_Res**2 ) )**0.5 )/ screen_size

    # query is out input to the model
    query= np.array([company, type, ram, weight, touchscreen, Ips, ppi, cpu, hdd, ssd, gpu, os])

    query= query.reshape(1, 12)

    # Since we transformed the Price using log transformation to Normally distribute it.
    # Therefore before predicting the price we will apply exponent on it.

    output= np.exp(pipe.predict(query))

    # Displaying the output using title function in the streamlit

    # The output is like [ 50848.64586 ] --> This is a list. We need to display its 0th element
    #                                           AND
    # It is better to convert the Price into int than showing floating digits which is equal to ( fils, cents)

    output= str (int( output[0] ))
    st.title("Predicted Price : " + output)





