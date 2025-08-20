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

from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

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



def standings_rank(team_dict, ranking):
    for date, key in team_dict.items():
        for item in key.keys():
            for x in team_dict[date][item]:
                idx = ranking.index(item) + 1
                team_dict[date][item][x]['Rank'] = idx
    return team_dict

# ## Shape of Data : 
# - box scores of 2 games vs opponent
# - 3 rows of 0s for padding
# - 10 recent games
# - 3 rows of 0s for padding
# - roster season stats(This is cheating right now because it has the whole season stats(TODO: write aggregation function to get season stats up to this point)




min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1,1))


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


def flatten_arrays(x_list):
    """Flatten the arrays for model input."""
    flattened_arrays = []
    for game_data in x_list:
        flattened_game = []
        for team_data in game_data:
            flattened_game.extend(team_data.flatten())  # Flattening each team's DataFrame
        flattened_arrays.append(flattened_game)
    return flattened_arrays



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



