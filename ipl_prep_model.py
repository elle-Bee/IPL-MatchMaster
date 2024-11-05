import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

match = pd.read_csv('/content/matches.csv')
delivery = pd.read_csv('/content/deliveries.csv')
home_team = pd.read_csv('/content/teamwise_home_and_away.csv')

match.head(4)

delivery.head(4)

delivery.columns

home_team.head(4)

match.shape

delivery.shape

total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df.rename(columns = {'total_runs':'Target'}, inplace = True)
total_score_df['Target'] = total_score_df['Target'].apply(lambda x: x+1)

total_score_df

match_df = match.merge(total_score_df[['match_id','Target']],left_on='id',right_on='match_id')

match_df

match_df['team1'].unique()

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df['toss_winner'] = match_df['toss_winner'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['toss_winner'] = match_df['toss_winner'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df = match_df[match_df['dl_applied'] == 0]

match_df = match_df[['match_id','city','winner','Target','toss_winner', 'toss_decision']]

match_df.shape

delivery_df = match_df.merge(delivery,on='match_id')

delivery_df.batting_team.value_counts()

delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Delhi Daredevils','Delhi Capitals')

delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

delivery_df = delivery_df[delivery_df['inning'] == 2]

delivery_df.shape

delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs']

delivery_df['runs_left'] = delivery_df['Target'] - delivery_df['current_score']

delivery_df['balls_left'] = 120 - ((delivery_df['over']-1)*6 + delivery_df['ball'])

delivery_df

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == 0 else 1)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['fallen_wickets'] = wickets
delivery_df.head()

delivery_df.shape

delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']

def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(result,axis=1)

final_df = delivery_df[['batting_team','bowling_team','city','toss_winner','toss_decision','runs_left','balls_left','fallen_wickets','Target','crr','rrr','result']]



final_df.batting_team.value_counts()

final_df

final_df = final_df.sample(final_df.shape[0])

home_team

home_team.iloc[3]['home_win_percentage'] += ((home_team.iloc[3]['home_wins'] + home_team.iloc[10]['home_wins'])*100)/(home_team.iloc[3]['home_matches'] + home_team.iloc[10]['home_matches'])
home_team.iloc[3]['away_win_percentage'] += ((home_team.iloc[3]['away_wins'] + home_team.iloc[10]['away_wins'])*100)/(home_team.iloc[3]['away_matches'] + home_team.iloc[10]['away_matches'])


home_team.iloc[4]['home_win_percentage'] += ((home_team.iloc[4]['home_wins'] + home_team.iloc[6]['home_wins'])*100)/(home_team.iloc[4]['home_matches'] + home_team.iloc[6]['home_matches'])
home_team.iloc[4]['away_win_percentage'] += ((home_team.iloc[4]['away_wins'] + home_team.iloc[6]['away_wins'])*100)/(home_team.iloc[4]['away_matches'] + home_team.iloc[6]['away_matches'])

home_team = home_team[home_team['team'].isin(teams)]

home_team.columns

home_team.reset_index()

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

def team_city_fix(row):
  t1 = team_city[row['batting_team']]
  t2 = team_city[row['bowling_team']]
  if row['city'] == t1:
     return 1
  elif row['city'] == t2:
    return 0
  else:
    return 2

final_df['Home_team'] = 0

final_df['Home_team'] = final_df.apply(team_city_fix,axis=1)

final_df.Home_team

hwp_lst = list(home_team.home_win_percentage)
awp_lst = list(home_team.away_win_percentage)
teams = list(home_team.team)

hwp = {}
awp = {}
for key in teams:
    for value in hwp_lst:
        hwp[key] = value
        hwp_lst.remove(value)
        break 

for key1 in teams:
    for value1 in awp_lst:
        awp[key1] = value1
        awp_lst.remove(value1)
        break

hwp

def home_p0(row):
  return hwp[row['bowling_team']]

def home_p1(row):
  return hwp[row['batting_team']]

def away_p0(row):
  return awp[row['bowling_team']]

def away_p1(row):
  return awp[row['batting_team']]

final_df['bowl_hwp'] = 0
final_df['bat_hwp'] = 0
final_df['bowl_awp'] = 0
final_df['bat_awp'] = 0

final_df['bowl_hwp'] = final_df.apply(home_p0,axis=1)
final_df['bat_hwp'] = final_df.apply(home_p1,axis=1)
final_df['bowl_awp'] = final_df.apply(away_p0,axis=1)
final_df['bat_awp'] = final_df.apply(away_p1,axis=1)

final_df.sample(3)

final_df.isnull().sum()

final_df.describe()

final_df.dropna(inplace=True)

final_df = final_df[final_df['balls_left'] != 0]

X = final_df.drop(['result'], axis=1)
y = final_df['result']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train

X_train.columns

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore'),['batting_team','bowling_team','city', 'toss_winner', 'toss_decision'])
]
,remainder='passthrough')

pipe1 = Pipeline(steps=[
    ('step1',trf),
    ('step2',XGBClassifier(random_state=42))
])

pipe1.fit(X_train,y_train)

y_pred1 = pipe1.predict(X_test)
accuracy_score(y_test,y_pred1)

pipe1.predict_proba(X_test)[99]

X_test.iloc[100]

pipe2 = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe2.fit(X_train,y_train)

y_pred2 = pipe2.predict(X_test)
accuracy_score(y_test,y_pred2)

pipe2.predict_proba(X_test)[99]

pickle.dump(pipe2,open('pipe.pkl','wb'))
pickle.dump(hwp,open('hwp.pkl','wb'))
pickle.dump(awp,open('awp.pkl','wb'))

