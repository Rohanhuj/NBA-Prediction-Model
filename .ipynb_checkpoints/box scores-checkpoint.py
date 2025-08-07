#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import datetime
import json
import optuna

import numpy as np
import pandas as pd
import xgboost as xgb

from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_importance
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from matplotlib import pyplot


# # Loading Scraped Data
# ### Team Dicts that are being loaded in pkl include daily box scores for team

# In[2]:


with open('dict15-16.pkl', 'rb') as f:
    team15_16 = pickle.load(f)


# In[3]:


with open('dict17-18.pkl', 'rb') as f:
    team17_18 = pickle.load(f)


# In[4]:


with open('dict18-19.pkl', 'rb') as f:
    team18_19 = pickle.load(f)


# In[5]:


with open('test23.pkl', 'rb') as f:
    test = pickle.load(f)


# In[6]:


with open('dict20-21.pkl', 'rb') as f:
    team20_21 = pickle.load(f)


# In[7]:


with open('dict21-22.pkl', 'rb') as f:
    team21_22 = pickle.load(f)


# In[8]:


with open('dict16-17.pkl', 'rb') as f:
    team16_17 = pickle.load(f)


# In[25]:


with open('team23_24_index_cleaned.pkl', 'rb') as f:
    team23_24 = pickle.load(f)


# In[30]:


with open('powerrankings15-16.pkl', 'rb') as f:
    ranking15_16 = pickle.load(f)


# In[31]:


with open('powerrankings17-18.json') as f:
    ranking17_18 = json.load(f)

with open('powerrankings18-19.json') as f:
    ranking18_19 = json.load(f)

with open('powerrankings20-21.json') as f:
    ranking20_21 = json.load(f)

with open('powerrankings21-22.json') as f:
    ranking21_22 = json.load(f)

with open('powerrankings22-23.json') as f:
    test_ranking = json.load(f)

with open('powerrankings16-17.pkl', 'rb') as f:
    ranking16_17 = pickle.load(f)

with open('powerrankings23-24.json', 'rb') as f:
    ranking23_24 = json.load(f)


# In[32]:


'''stats for every player in the NBA'''
test_df = pd.read_excel("totalstats22-23.xlsx")
df21_22 = pd.read_excel("totalstats21-22.xlsx")
df20_21 = pd.read_excel("totalstats20-21.xlsx")
df18_19 = pd.read_excel("totalstats18-19.xlsx")
df17_18 = pd.read_excel("totalstats17-18.xlsx")
df16_17 = pd.read_excel("totalstats16-17.xlsx")
df15_16 = pd.read_excel("totalstats15-16.xlsx")


# In[13]:


team_list = list(test_df.Tm.unique())


# In[188]:


label_encoder = LabelEncoder()
label_encoder.fit(team_list)

# Map team name to encoded integer
team_encoder = {team: label_encoder.transform([team])[0] for team in team_list}

# (Optional) If you ever need reverse mapping:
team_decoder = {v: k for k, v in team_encoder.items()}


# In[33]:


from sklearn import preprocessing

def encode(team_dict):
    """Generate player and team encoders, and safe encode functions."""
    player_set = set()
    team_set = set()

    for date in team_dict:
        for team1 in team_dict[date]:
            team_set.add(team1)
            for team2 in team_dict[date][team1]:
                team_set.add(team2)
                df1 = team_dict[date][team1][team2]
                df2 = team_dict[date][team2][team1]

                for df in [df1, df2]:
                    if 'Starters' in df.columns:
                        starters = df['Starters'].iloc[:-1]
                        player_set.update(starters)

    # Create encoders
    player_encoder = preprocessing.LabelEncoder()
    team_encoder = preprocessing.LabelEncoder()

    player_encoder.fit(list(player_set))
    team_encoder.fit(list(team_set))

    # Safe encoding functions
    def safe_team_encode(team_name, fallback=None):
        if isinstance(team_name, str) and team_name in team_encoder.classes_:
            return team_encoder.transform([team_name])[0]
        elif fallback and fallback in team_encoder.classes_:
            return team_encoder.transform([fallback])[0]
        else:
            return -1

    def safe_player_encode(player_name):
        if isinstance(player_name, str) and player_name != "Team Totals" and player_name in player_encoder.classes_:
            return player_encoder.transform([player_name])[0]
        return -1

    return player_encoder, team_encoder, safe_player_encode, safe_team_encode


# In[88]:


pd.set_option('display.max_rows', None)


# In[34]:


def convert_MP(mp):
    if(mp != 0 and not(isinstance(mp, float)) and not(isinstance(mp, int))):
        time = mp.split(':')
        mp = int(time[0]) * 60 + int(time[1])

    elif(isinstance(mp, float)):
        mp = 60 * mp

    return mp


# In[35]:


def process_dict(team_dict, safe_player_encode, safe_team_encode):
    for date in team_dict:
        for home in team_dict[date]:
            for away in team_dict[date][home]:
                df = team_dict[date][home][away]
                df = df.drop([5])
                df = df.reset_index(drop=True)
                df = df.fillna(0)
                df = df.replace(['Did Not Play', 'Did Not Dress', 'Not With Team', 'Player Suspended'], 0)

                # Apply safe team encoding
                df['Team'] = safe_team_encode(home)

                # Apply safe player encoding
                if 'Starters' in df.columns:
                    df['PlayerID'] = df['Starters'].apply(
                        lambda name: safe_player_encode(name) if isinstance(name, str) and name != "Team Totals" else name
                    )

                # Move target row
                target_row = df.iloc[-1]
                df = df.drop(df.index[-1])
                df = df.reindex(range(12), fill_value=0)
                df.loc[len(df.index)] = target_row

                team_dict[date][home][away] = df

    return team_dict


# In[36]:


player_enc20_21, team_enc20_21, safeplayers20_21, safeteam20_21 = encode(team20_21)
player_enc21_22, team_enc21_22, safeplayers21_22, safeteam21_22 = encode(team21_22)
player_enc18_19, team_enc18_19, safeplayers18_19, safeteam18_19 = encode(team18_19)
player_enc17_18, team_enc17_18, safeplayers17_18, safeteam17_18 = encode(team17_18)
player_enc16_17, team_enc16_17, safeplayers16_17, safeteam16_17 = encode(team16_17)
player_enc15_16, team_enc15_16, safeplayers15_16, safeteam15_16 = encode(team15_16)
playerenctest, teamenctest, safeplayerstest, safeteamtest = encode(test)


# In[37]:


team_dict20_21 = process_dict(team20_21, safeplayers20_21, safeteam20_21)
team_dict21_22 = process_dict(team21_22, safeplayers21_22, safeteam21_22)
team_dict18_19 = process_dict(team18_19, safeplayers18_19, safeteam18_19)
team_dict17_18 = process_dict(team17_18, safeplayers17_18, safeteam17_18)
team_dict16_17 = process_dict(team16_17, safeplayers16_17, safeteam16_17)
team_dict15_16 = process_dict(team15_16, safeplayers15_16, safeteam15_16)
test = process_dict(test, safeplayerstest, safeteamtest)


# In[213]:


df = pd.DataFrame.from_dict(team_dict20_21, orient='index')
team_data = df.loc['02-05-2021', 'PHO']
team_data['DET']


# In[165]:


df = pd.DataFrame.from_dict(test, orient='index')
team_data = df.loc['02-01-2023', 'ORL']
team_data['PHI']


# In[84]:


def find_recent_games(team_dict, team, date, end_date, games = 10):
    last_games = pd.DataFrame()
    #end_date = datetime.date(2022, 10, 17)        
    delta = datetime.timedelta(days = -1)
    count = 0
    while(date > end_date):
        day = str(date.day)
        month = str(date.month)
        year = str(date.year)
        if(date.day < 10):
            day = "0" + day
        if(date.month < 10):
            month = "0" + month
        loop_date = '' + month + '-' + day + '-' + year  
        for i, key in team_dict.items():
            if(i == loop_date):
                for item in team_dict[loop_date].keys():
                    if(item == team):
                        for x in team_dict[loop_date][item]:
                            last_games = pd.concat([last_games, team_dict[loop_date][item][x]])
                            count += 1
                            if(count >= games):
                                return last_games
                        #last_games[x] = team_dict[loop_date][item][x]
        date += delta
    if(count <= games):
        return pd.DataFrame()
    #last_games = dict(list(last_games.items())[:games])
    return last_games


# In[39]:


def get_games_against_opponent(team_dict, team, opponent, date, start_date):
    selected_team = pd.DataFrame()
    #opposing_team = []
    #start_date = datetime.date(2022, 10, 17)
    delta = datetime.timedelta(days = -1)
    date += delta
    count = 0

    while(date > start_date):
        day = str(date.day)
        month = str(date.month)
        year = str(date.year)
        if(date.day < 10):
            day = "0" + day
        if(date.month < 10):
            month = "0" + month
        loop_date = '' + month + '-' + day + '-' + year
        for i, key in team_dict.items():
            if(i == loop_date):
                for item in team_dict[loop_date].keys():
                    if(item == team):
                        for x in team_dict[loop_date][item]:
                            if(x == opponent):
                                selected_team = pd.concat([selected_team, team_dict[loop_date][item][x]])
                                #opposing_team.append(team_dict[date][x][item])
                                count += 1
                                if(count >= 2):
                                    return selected_team

        date += delta
    if(count == 1):
        selected_team = pd.concat([selected_team, selected_team])
    if(count == 0):
        return pd.DataFrame()
    return selected_team 


# In[40]:


def process_seasondf(df, safe_player_encode, safe_team_encode):
    df.rename(columns = {'Player' : 'Starters', 'OFFRTG' : 'ORtg', 'DEFRTG' : 'DRtg'}, inplace = True)

    df_rosters = {}
    team_list = list(df.Tm.unique())
    tms = df.groupby('Tm')

    df_sort = team21_22['10-30-2021']['CLE']['PHO']

    for team, groupdf in tms:
        groupdf = groupdf.sort_values(by = 'MP', ascending = False)
        groupdf = groupdf.reindex(df_sort.columns, axis = 1)
        groupdf = groupdf.head(12)

        groupdf['Team'] = safe_team_encode(team)
        groupdf['PlayerID'] = groupdf['Starters'].apply(lambda name: safe_player_encode(name))

        df_rosters[team] = groupdf

    return df_rosters


# In[41]:


def shape_data(team_dict, df_rosters, end_date):

    arrays = []
    game_keys = []
    fullgamedata = {}
    processed_matchups = set()

    for date, daily_games in team_dict.items():
        for team1 in daily_games.keys():
            for team2 in team_dict[date][team1]:

                matchup_key = tuple(sorted([team1, team2]) + [date])
                if matchup_key in processed_matchups:
                    continue
                processed_matchups.add(matchup_key)

                form = '%m-%d-%Y'
                new_date = datetime.datetime.strptime(date, form).date()

                if(get_games_against_opponent(team_dict, team1, team2, new_date, end_date).empty or 
                    find_recent_games(team_dict, team1, new_date, end_date).empty or
                    get_games_against_opponent(team_dict, team2, team1, new_date, end_date).empty or 
                    find_recent_games(team_dict, team2, new_date, end_date).empty):
                    print('not enough games')
                    continue


                team12 = get_games_against_opponent(team_dict, team1, team2, new_date, end_date)
                team1recent = find_recent_games(team_dict, team1, new_date, end_date)

                team21 = get_games_against_opponent(team_dict, team2, team1, new_date, end_date)
                team2recent = find_recent_games(team_dict, team2, new_date, end_date)

                pad_df = pd.DataFrame(0, index = range(3), columns = team1recent.columns)

                df = pd.concat([team12, pad_df, team1recent, pad_df, team21, pad_df, team2recent, pad_df])
                df = df.drop(['Home'], axis = 1)
                arrays.append(df)
                game_keys.append((date, team1, team2))

                if(float(team_dict[date][team1][team2].loc[12]['PTS']) > float(team_dict[date][team2][team1].loc[12]['PTS'])):
                    win = 1
                    actual_winner = team1
                else:
                    win = 0
                    actual_winner = team2

                score = (float(team_dict[date][team1][team2].loc[12]['PTS']), float(team_dict[date][team2][team1].loc[12]['PTS']))

                fullgamedata[(date, team1, team2)] = {
                    "team1" : team1,
                    "team2" : team2,
                    "label" : win,
                    "date" : date,
                    "score" : score,
                    "actual_winner" : actual_winner
                }
                print(team1 + ' vs ' + team2 + ' winner: ' + actual_winner + "score " + str(score) + ' complete')
        print('' + str(date) + ' day done')

    return arrays, game_keys, fullgamedata


# In[42]:


df_roster20_21 = process_seasondf(df20_21, safeplayers20_21, safeteam20_21)
df_roster21_22 = process_seasondf(df21_22, safeplayers21_22, safeteam21_22)
df_roster18_19 = process_seasondf(df18_19, safeplayers18_19, safeteam18_19)
df_roster17_18 = process_seasondf(df17_18, safeplayers17_18, safeteam17_18)
df_roster16_17 = process_seasondf(df16_17, safeplayers16_17, safeteam16_17)
df_roster15_16 = process_seasondf(df15_16, safeplayers15_16, safeteam15_16)
test_df = process_seasondf(test_df, safeplayerstest, safeteamtest)


# In[43]:


def power_rank(ranking, team_dict, end_date):

    keylist = list(ranking.keys())

    for date, key in team_dict.items():
        for i, v in enumerate(keylist):
            form = '%m-%d-%Y'
            new_date = datetime.datetime.strptime(date, form).date()
            try:
                dateright = keylist[i + 1]
            except:
                dateright = end_date
            dateright = datetime.datetime.strptime(dateright, form).date()
            dateleft = datetime.datetime.strptime(v, form).date()
            if(new_date >= dateleft and new_date < dateright):
                rankingdate = dateleft
            for item in key.keys():
                for x in team_dict[date][item]:
                    output_form = '%m-%d-%Y'
                    rank_date = datetime.datetime.strftime(rankingdate, output_form)
                    idx = ranking[rank_date].index(item) + 1
                    team_dict[date][item][x]['Rank'] = idx
    return team_dict


# In[44]:


def standings_rank(team_dict, ranking):
    for date, key in team_dict.items():
        for item in key.keys():
            for x in team_dict[date][item]:
                idx = ranking.index(item) + 1
                team_dict[date][item][x]['Rank'] = idx
    return team_dict


# In[45]:


team_dict20_21 = power_rank(ranking20_21, team_dict20_21, '05-16-2021')
team_dict21_22 = power_rank(ranking21_22, team_dict21_22, '04-10-2022')
team_dict21_22 = power_rank(ranking21_22, team_dict21_22, '04-10-2022')
team_dict18_19 = power_rank(ranking18_19, team_dict18_19, '04-10-2019')
team_dict17_18 = power_rank(ranking17_18, team_dict17_18, '04-11-2018')
test = power_rank(test_ranking, test, '04-09-2023')


# In[46]:


team_dict16_17 = standings_rank(team_dict16_17, ranking16_17)
team_dict15_16 = standings_rank(team_dict15_16, ranking15_16)


# In[47]:


arrays18_19, keys18_19, data18_19 = shape_data(team_dict18_19, df_roster18_19, datetime.date(2018, 10, 15))
arrays17_18, keys17_18, data17_18 = shape_data(team_dict17_18, df_roster17_18, datetime.date(2017, 10, 16))
testarrays, testkeys, testdata = shape_data(test, test_df, datetime.date(2022, 10, 17))
arrays21_22, keys21_22, data21_22 = shape_data(team_dict21_22, df_roster21_22, datetime.date(2021, 10, 18))
arrays20_21, keys20_21, data20_21 = shape_data(team_dict20_21, df_roster20_21, datetime.date(2020, 10, 21))
arrays16_17, keys16_17, data16_17 = shape_data(team_dict16_17, df_roster16_17, datetime.date(2016, 10, 24))
arrays15_16, keys15_16, data15_16 = shape_data(team_dict15_16, df_roster15_16, datetime.date(2015, 10, 26))


# ## Shape of Data : 
# - box scores of 2 games vs opponent
# - 3 rows of 0s for padding
# - 10 recent games
# - 3 rows of 0s for padding
# - roster season stats(This is cheating right now because it has the whole season stats(TODO: write aggregation function to get season stats up to this point)

# In[48]:


min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1,1))


# In[49]:


def final_shape(arrays):
    x_list = []
    for array in arrays:
        array = array[array['Starters'] != 'Team Totals']
        array = array.drop(['Starters'], axis = 1)
        #array = array.drop(['Home'], axis = 1)
        array['MP'] = array.loc[:]['MP'].apply(convert_MP)
        array = array.fillna(0)
        array = array.astype(float)
        array = min_max_scaler.fit_transform(array)
        x_list.append(array)
    return x_list


# In[50]:


x_test = final_shape(testarrays)
x20_21 = final_shape(arrays20_21)
x21_22 = final_shape(arrays21_22)
x18_19 = final_shape(arrays18_19)
x17_18 = final_shape(arrays17_18)
x16_17 = final_shape(arrays16_17)
x15_16 = final_shape(arrays15_16)


# In[89]:


testarrays[0]


# In[51]:


y21_22 = [data21_22[key]['label'] for key in keys21_22]
y20_21 = [data20_21[key]['label'] for key in keys20_21]
y18_19 = [data18_19[key]['label'] for key in keys18_19]
y17_18 = [data17_18[key]['label'] for key in keys17_18]
y16_17 = [data16_17[key]['label'] for key in keys16_17]
y15_16 = [data15_16[key]['label'] for key in keys15_16]
y_test =  [testdata[key]['label'] for key in testkeys]


# In[52]:


len(x_test)


# In[53]:


len(y_test)


# In[54]:


missing_keys = [key for key in testkeys if key not in testdata]
print(f"Missing keys: {len(missing_keys)}")


# In[55]:


y_list = y15_16 + y16_17 + y17_18 + y18_19 + y20_21 + y21_22


# In[56]:


x_list = x21_22 + x20_21 + x18_19 + x17_18 + x16_17 + x15_16


# In[57]:


keys = keys21_22 + keys20_21 + keys18_19 + keys17_18 + keys16_17 + keys15_16


# In[58]:


flattened_arrays = []
for game_data in x_list:
    flattened_game = []
    for team_data in game_data:
        flattened_game.extend(team_data.flatten())  # Flattening each team's DataFrame
    flattened_arrays.append(flattened_game)


# In[59]:


X = np.array(flattened_arrays)


# In[60]:


Y = np.array(y_list)


# In[61]:


X.shape


# In[62]:


Y.shape


# In[63]:


x_test = np.array(x_test)


# In[64]:


x_test.shape


# In[65]:


x_test = x_test.reshape(x_test.shape[0], -1)


# In[66]:


x_test.shape


# In[67]:


y_test = np.array(y_test)


# In[68]:


y_test.shape


# In[69]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.1)


# In[70]:


eval_set = [(X_train, Y_train), (X_test, Y_test)]
evals_result = {}


# In[75]:


def objective(trial, data = X_train, target = Y_train):

    param = {
        'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 1.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': trial.suggest_categorical('n_estimators', [2000, 3000, 3500, 4000, 5000, 6000]),
        'max_depth': trial.suggest_categorical('max_depth', [1, 3, 5,7,9,11,13,15,17]),
        #'random_state': trial.suggest_categorical('random_state', [2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
    }
    model = xgb.XGBClassifier(**param)  

    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)

    preds = model.predict(x_test)

    accuracy = accuracy_score(wins_test, preds)

    return accuracy


# In[101]:


def objective(trial):
    # Suggested hyperparameters
    param = {
        'tree_method': 'gpu_hist',
        'eval_metric': 'logloss',
        'lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 4000, step=500),
        'max_depth': trial.suggest_int('max_depth', 3, 15, step=2),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    }

    model = xgb.XGBClassifier(**param)

    model.fit(
    X=X_train,
    y=Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=False
    )

    preds = model.predict(X_test)
    accuracy = accuracy_score(Y_test, preds)
    return accuracy


# In[102]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials = 300)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
trial = study.best_trial
print("  Value: ", trial.value)


# In[103]:


#best_params = study.best_trial.params
params = {'lambda': 0.34649586581992475, 'alpha': 0.418230742143025, 'colsample_bytree': 0.8, 'subsample': 0.8, 'learning_rate': 0.016, 'n_estimators': 2000, 'max_depth': 17, 'min_child_weight': 2, 'gamma': 6.422333220506358e-08}


# In[71]:


model = xgb.XGBClassifier(n_estimators = 1000, learning_rate = 0.01, max_depth = 3, subsample = 0.7, colsample_bytree = 0.7, 
                          reg_alpha = 1, reg_lambda = 2, min_child_weight = 5, 
                          tree_method = 'gpu_hist', eval_metric=["error", "logloss"], feature_selector = 'shuffle')


# In[72]:


model.fit(X_train, Y_train, eval_set = eval_set, verbose=True)


# In[53]:


results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(epochs)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train Log Loss')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation Log Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[54]:


plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['error'], label='Train Error')
plt.plot(x_axis, results['validation_1']['error'], label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[55]:


train_acc = [1 - e for e in results['validation_0']['error']]
val_acc = [1 - e for e in results['validation_1']['error']]

plt.figure(figsize=(10, 5))
plt.plot(x_axis, train_acc, label='Train Accuracy')
plt.plot(x_axis, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('XGBoost Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[73]:


predictions = model.predict(x_test)


# In[57]:


len(testkeys)


# In[75]:


results = []
for i, key in enumerate(testkeys):
    date, team1, team2 = key
    predicted = predictions[i]
    actual = testdata[key]['label']

    pred_winner = team1 if predicted == 1 else team2
    actual_winner = testdata[key]['actual_winner']
    score = testdata[key]['score']
    correct = int(predicted == actual)

    results.append({
        "Date" : date,
        "Team 1" : team1,
        "Team 2" : team2,
        "Score" : score,
        "Predicted Winner" : pred_winner,
        "Actual Winner" : actual_winner,
        "Correct Prediction" : 1 if correct else 0
    })

results_df = pd.DataFrame(results)


# In[76]:


results_df


# In[65]:


results_df.to_csv('results.csv')


# In[77]:


results_df['Score_Diff'] = results_df['Score'].apply(lambda x: abs(x[0] - x[1]))

correct_diff = results_df[results_df['Correct Prediction'] == 1]['Score_Diff'].mean()
incorrect_diff = results_df[results_df['Correct Prediction'] == 0]['Score_Diff'].mean()
print(correct_diff)
print(incorrect_diff)


# In[78]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[139]:


precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# In[99]:


feature_importance = model.feature_importances_
print(feature_importance)


# In[100]:


score_dict = model.get_booster().get_score(importance_type='gain')


# In[101]:


scores_list = list(score_dict.keys())
scores_list


# In[102]:


col_list = []
for col in arrays21_22[0].columns:
    if col != 'Starters':
        col_list.append(col)

col_list


# In[103]:


count = 0
sco = {}

for i in range(37):
    sco[i] = []


for key, val in score_dict.items():
    if(count >= 37):
        count = 0
    sco[count].append(val)
    count += 1

features = {}
for new_key, (_, value) in zip(col_list, sco.items()):
    features[new_key] = value

averages = {}
for col, percent in features.items():
    averages[col] = mean(percent)

sorted_data_descending = dict(sorted(averages.items(), key=lambda item: item[1], reverse=True))
sorted_data_descending


# In[136]:


results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)


# In[137]:


fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()


# In[138]:


fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()


# In[29]:


'''
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=(5), activation="relu", input_shape=(164,35)))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))
model.add(Conv1D(filters=32, kernel_size=(5), activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=(5), activation="relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=(5), activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
'''


# In[30]:


#opt = keras.optimizers.Adam(learning_rate=0.001)
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[31]:


#model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size=128)


# In[113]:


#train_loss, train_accuracy = model.evaluate(X_train, y_train)

#print(f"Training Accuracy: {train_accuracy:.4f}")


# In[ ]:


results = []

for i, key in enumerate(key_test):  # key = (date, team1, team2)
    predicted = predictions[i]
    actual = y_test[i]

    team1 = key[1]
    team2 = key[2]
    date = key[0]

    predicted_winner = team1 if predicted == 1 else team2
    actual_winner = game_meta[key]["actual_winner"]
    correct = predicted_winner == actual_winner

    results.append({
        "Date": date,
        "Team 1": team1,
        "Team 2": team2,
        "Predicted Winner": predicted_winner,
        "Actual Winner": actual_winner,
        "Correct Prediction": 1 if correct else 0
    })

results_df = pd.DataFrame(results)
results_df["Date"] = pd.to_datetime(results_df["Date"])
results_df = results_df.sort_values("Date").reset_index(drop=True)


# In[86]:


def predict_single_game(model, team_dict, team1, team2, game_date, end_date):
    """
    Predicts the winner between two teams on a given date using your trained model.
    """

    game_date = datetime.datetime.strptime(game_date, "%m-%d-%Y").date()

    player_enc, team_enc, safeplayers, safeteam = encode(team_dict)
    new_dict = process_dict(team_dict, safeplayers, safeteam)

    # Step 1: Pull recent game data
    recent_team1 = find_recent_games(new_dict, team1, game_date, end_date)
    recent_team2 = find_recent_games(new_dict, team2, game_date, end_date)
    vs_team1 = get_games_against_opponent(new_dict, team1, team2, game_date, end_date)
    vs_team2 = get_games_against_opponent(new_dict, team2, team1, game_date, end_date)

    if recent_team1.empty or recent_team2.empty or vs_team1.empty or vs_team2.empty:
        print("Not enough historical data for this matchup.")
        return

    # Step 2: Assemble data
    pad = pd.DataFrame(0, index=range(3), columns=recent_team1.columns)
    df = pd.concat([
        vs_team1, pad,
        recent_team1, pad,
        vs_team2, pad,
        recent_team2, pad,
    ])

    df = df[df['Starters'] != 'Team Totals']
    df = df.drop(['Starters', 'Home'], axis=1)
    df['MP'] = df['MP'].apply(convert_MP)
    df = df.fillna(0).astype(float)
    df = min_max_scaler.fit_transform(df)

    # Step 3: Flatten for model
    flattened = []
    for row in df:
        flattened.extend(row.flatten())
    input_array = np.array(flattened).reshape(1, -1)

    # Step 4: Predict
    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]
    confidence = prob[pred]

    predicted_winner = team1 if pred == 1 else team2

    print(f"Prediction: {predicted_winner}")
    print(f"Confidence: {confidence:.2%}")


# In[87]:


predict_single_game(model, team23_24, 'OKC', 'PHI', '04-02-2024', datetime.date(2023, 10, 24))


# In[ ]:


arrays = []
game_keys = []
fullgamedata = {}
processed_matchups = set()

for date, daily_games in team_dict.items():
    for team1 in daily_games.keys():
        for team2 in team_dict[date][team1]:

            matchup_key = tuple(sorted([team1, team2]) + [date])
            if matchup_key in processed_matchups:
                continue
            processed_matchups.add(matchup_key)

            form = '%m-%d-%Y'
            new_date = datetime.datetime.strptime(date, form).date()

            if(get_games_against_opponent(team_dict, team1, team2, new_date, end_date).empty or 
                find_recent_games(team_dict, team1, new_date, end_date).empty or
                get_games_against_opponent(team_dict, team2, team1, new_date, end_date).empty or 
                find_recent_games(team_dict, team2, new_date, end_date).empty):
                print('not enough games')
                continue


# In[ ]:


from datetime import date

predict_single_game(
    model=model,
    team_dict=team_dict23_24,
    team1="CHI",
    team2="BOS",
    game_date = "02-22-2024",
    end_date=date(2023, 10, 24)  # start of season
)


# In[120]:


arrays_to_process = [x21_22, x18_19, x17_18, x16_17, x15_16]

# Determine the minimum number of samples across all arrays
min_samples = min(len(arr) for arr in arrays_to_process)

# Drop data points to make all arrays the same shape
for idx, array in enumerate(arrays_to_process):
    arrays_to_process[idx] = array[:min_samples]

# Concatenate the processed arrays into a single array
X_combined = np.concatenate(arrays_to_process, axis=0)

X_combined.shape


# In[121]:


wins_to_process = [wins21_22, wins18_19, wins17_18, wins16_17, wins15_16]

# Determine the minimum number of samples across all 'wins' arrays
min_samples = min(len(wins) for wins in wins_to_process)

# Drop data points to make all 'wins' arrays the same shape
for idx, wins_array in enumerate(wins_to_process):
    wins_to_process[idx] = wins_array[:min_samples]

# Concatenate the processed 'wins' arrays into a single array
y_combined = np.concatenate(wins_to_process, axis=0)

y_combined.shape


# In[ ]:


recent_team1 = find_recent_games(team_dict, team1, game_date, end_date)


# In[118]:


import numpy as np
from collections import Counter

print("Train label distribution:", Counter(y_train))
print("Test label distribution:", Counter(y_test))
print("Train label mean (should be near 0.5):", np.mean(y_train))
print("Test label mean:", np.mean(y_test))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[124]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
print("LR Accuracy:", lr.score(X_test, Y_test))


# In[127]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
print("Cross-validated accuracy:", scores.mean())


# In[125]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_vis = PCA(n_components=2).fit_transform(X_train)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap='coolwarm', alpha=0.6)
plt.title("PCA of Training Data")
plt.show()


# In[ ]:




