import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

team_city = {
    'Sunrisers Hyderabad' : 'Hyderabad',
    'Mumbai Indians' : 'Mumbai',
    'Royal Challengers Bangalore' : 'Bangalore',
    'Kolkata Knight Riders' : 'Kolkata',
    'Kings XI Punjab' : 'Chandigarh',
    'Chennai Super Kings' : 'Chennai',
    'Rajasthan Royals' : 'Jaipur',
    'Delhi Capitals' : 'Delhi'
}

pipe = pickle.load(open('pipe.pkl','rb'))
hwp = pickle.load(open('hwp.pkl', 'rb'))
awp = pickle.load(open('awp.pkl', 'rb'))
st.title('IPL Win Predictor')
st.image("https://wallpapercave.com/w/wp2458583.jpg")


col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting teamr',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

col3, col4 = st.columns(2)

with col3:
    toss_winner = st.selectbox('Toss winner',sorted([batting_team, bowling_team]))
with col4:
    toss_decision = st.selectbox('Toss decision',sorted(['field', 'bat']))

target = st.number_input('Target')

col5,col6,col7 = st.columns(3)

with col5:
    score = st.number_input('Score')
with col6:
    overs = st.number_input('Overs completed')
with col7:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    if overs == 0:
        st.write('**OVERS CANNOT BE ZERO**')

    else:
        runs_left = target - score
        balls_left = 120 - (overs*6)
        crr = score/overs
        rrr = (runs_left*6)/balls_left

        bowl_city = team_city[bowling_team]
        bat_city = team_city[batting_team]
        home_team = 0

        if selected_city == bowl_city:
            home_team = 0
        elif selected_city == bat_city:
            home_team = 1
        else:
            home_team = 2

        bowl_hwp  = hwp[bowling_team]
        bat_hwp = hwp[batting_team]
        bowl_awp = awp[bowling_team]
        bat_awp = awp[batting_team]

        input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'toss_winner':[toss_winner],'toss_decision':[toss_decision],'runs_left':[runs_left],'balls_left':[balls_left],'fallen_wickets':[wickets],'Target':[target],'crr':[crr],'rrr':[rrr], 'Home_team':[home_team], 'bowl_hwp':[bowl_hwp], 'bat_hwp':[bat_hwp], 'bowl_awp':[bowl_awp], 'bat_awp':[bat_awp]})

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(batting_team + "- " + str(round(win*100)) + "%")
        st.header(bowling_team + "- " + str(round(loss*100)) + "%")