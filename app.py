import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pickle
import numpy as np
import time


modelrfc = pickle.load(open(r"C:\Users\HP\Desktop\priyanka\OLYMPICS_DATA_ANALYSER_AND_PREDICTOR\modelrfc.pkl","rb"))

modellr = pickle.load(open(r"C:\Users\HP\Desktop\priyanka\OLYMPICS_DATA_ANALYSER_AND_PREDICTOR\modellr.pkl", "rb"))
#modelnn = pickle.load(open("modelnn.pkl","rb"))
transformer = pickle.load(open("transformer.pkl","rb"))

df = pd.read_csv(r"C:\Users\HP\Desktop\priyanka\OLYMPICS_DATA_ANALYSER_AND_PREDICTOR\athlete_events.csv")
region_df = pd.read_csv(r"C:\Users\HP\Desktop\priyanka\OLYMPICS_DATA_ANALYSER_AND_PREDICTOR\noc_regions.csv")
df = preprocessor.preprocess(df,region_df)

st.sidebar.title("OLYMPICS DATA ANALYSZER AND PREDICTOR")
st.sidebar.image("Olympics-Symbol.png")
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally',"Medal Predictor")
)


if user_menu == "Medal Tally":
    st.header("Medal Tally (From Athens 1896 to Rio 2016)")
    years,country = helper.country_year_list(df)
    selected_year = st.selectbox("Select Year",years)
    selected_country = st.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    medal_tally.rename(columns = {"region":"Country"},inplace=True)
    if selected_year == "Overall" and selected_country == "Overall":
        st.title("Overall Medal Tally")
    if selected_year != "Overall" and selected_country == "Overall":
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == "Overall" and selected_country != "Overall":
        st.title(selected_country + " overall performance in Olympics")
    if selected_year != "Overall" and selected_country != "Overall":
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")

    st.table(medal_tally)



if user_menu == "Medal Predictor":
    st.title("Olympics Medal Predictor")
    selected_col = ["Sex" , "region" ,"Sport","Height" , "Weight" , "Age" ]
    sport = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 'Curling', 'Cycling', 'Diving', 'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 'Gymnastics', 'Handball', 'Hockey', 'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol', 'Modern Pentathlon', 'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping', 'Snowboarding', 'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']
    country = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA', 'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']
    with st.form("my_form"):
        Sex = st.selectbox("Select Sex",["M","F"])
        Age = st.slider("Select Age",10,97)
        Height = st.slider("Select Height(In centimeters)",127,226)
        Weight = st.slider("Select Weight(In kilograms)",20,214)
        region = st.selectbox("Select Country",country)
        Sport = st.selectbox("Select Sport",sport)
        input_model = st.selectbox("Select Prediction Model",["Random Forest Classifier","Logistic Regression","Neutral Network"])


        
        submitted = st.form_submit_button("Submit")
        if submitted:
            inputs = [Sex,region,Sport,Height,Weight,Age]
            inputs = pd.DataFrame([inputs],columns=selected_col)
            inputs = transformer.transform(inputs)
            if input_model == "Random Forest Classifier":
                model = modelrfc
            if input_model == "Logistic Regression":
                model = modellr
            if input_model == "Neutral Network":
                model = modelrfc
            prediction = model.predict(inputs)
            #prediction = np.argmax(prediction[0])
            with st.spinner('Predicting output...'):
                time.sleep(1)
                if prediction[0] == 0 :
                    ans = "Low"
                    st.warning("Medal winning probability is {}".format(ans),icon="⚠️")
                else :
                    ans = "High"
                    st.success("Medal winning probability is {}".format(ans),icon="✅")

