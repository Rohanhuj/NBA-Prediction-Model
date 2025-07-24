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
from xgboost import plot_importance
from matplotlib import pyplot

with open('dict15-16.pkl', 'rb') as f:
    team15_16 = pickle.load(f)

with open('dict17-18.pkl', 'rb') as f:
    team17_18 = pickle.load(f)

with open('dict18-19.pkl', 'rb') as f:
    team18_19 = pickle.load(f)

with open('test23.pkl', 'rb') as f:
    test = pickle.load(f)

with open('dict20-21.pkl', 'rb') as f:
    team20_21 = pickle.load(f)

with open('dict21-22.pkl', 'rb') as f:
    team21_22 = pickle.load(f)

with open('dict16-17.pkl', 'rb') as f:
    team16_17 = pickle.load(f)

with open('powerrankings15-16.pkl', 'rb') as f:
    ranking15_16 = pickle.load(f)

with open('powerrankings16-17.pkl', 'rb') as f:
    ranking16_17 = pickle.load(f)

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

test_df = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats22-23.xlsx")
df21_22 = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats21-22.xlsx")
df20_21 = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats20-21.xlsx")
df18_19 = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats18-19.xlsx")
df17_18 = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats17-18.xlsx")
df16_17 = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats16-17.xlsx")
df15_16 = pd.read_excel("/home/rohan/python-projects/jupyter/BasketballProject/totalstats15-16.xlsx")

team_list = list(test_df.Tm.unique())

encoders = {}
label_encoder = preprocessing.LabelEncoder()
encoded_teams = label_encoder.fit(team_list)
encoded = label_encoder.transform(team_list)

for i in range(len(encoded)):
    team = label_encoder.inverse_transform(encoded)[i]
    encoders[team] = encoded[i]

pd.set_option('display.max_columns', None)


def convert_MP(mp):
    if (mp != 0 and not (isinstance(mp, float)) and not (isinstance(mp, int))):
        time = mp.split(':')
        mp = int(time[0]) * 60 + int(time[1])

    elif (isinstance(mp, float)):
        mp = 60 * mp

    return mp


def process_dict(team_dict):
    for date in team_dict:
        for home in team_dict[date]:
            for away in team_dict[date][home]:
                team_dict[date][home][away] = team_dict[date][home][away].drop([5])
                team_dict[date][home][away] = team_dict[date][home][away].reset_index(drop=True)
                team_dict[date][home][away] = team_dict[date][home][away].fillna(0)
                team_dict[date][home][away] = team_dict[date][home][away].replace('Did Not Play', 0)
                team_dict[date][home][away] = team_dict[date][home][away].replace('Did Not Dress', 0)
                team_dict[date][home][away] = team_dict[date][home][away].replace('Not With Team', 0)
                team_dict[date][home][away] = team_dict[date][home][away].replace('Player Suspended', 0)
                team_dict[date][home][away]['Team'] = encoders[home]
                target_row = team_dict[date][home][away].iloc[-1]
                team_dict[date][home][away] = team_dict[date][home][away].drop(team_dict[date][home][away].index[-1])
                team_dict[date][home][away] = team_dict[date][home][away].reindex(range(12), fill_value=0)
                team_dict[date][home][away].loc[len(team_dict[date][home][away].index)] = target_row

    return team_dict


team_dict20_21 = process_dict(team20_21)
team_dict21_22 = process_dict(team21_22)
team_dict18_19 = process_dict(team18_19)
team_dict17_18 = process_dict(team17_18)
team_dict16_17 = process_dict(team16_17)
team_dict15_16 = process_dict(team15_16)
test = process_dict(test)


def find_recent_games(team_dict, team, date, end_date, games=10):
    last_games = pd.DataFrame()
    # end_date = datetime.date(2022, 10, 17)
    delta = datetime.timedelta(days=-1)
    count = 0

    while (date > end_date):
        day = str(date.day)
        month = str(date.month)
        year = str(date.year)
        if (date.day < 10):
            day = "0" + day
        if (date.month < 10):
            month = "0" + month
        loop_date = '' + month + '-' + day + '-' + year
        for i, key in team_dict.items():
            if (i == loop_date):
                for item in team_dict[loop_date].keys():
                    if (item == team):
                        for x in team_dict[loop_date][item]:
                            last_games = pd.concat([last_games, team_dict[loop_date][item][x]])
                            count += 1
                            if (count >= games):
                                return last_games
                        # last_games[x] = team_dict[loop_date][item][x]
        date += delta
    if (count <= games):
        return pd.DataFrame()
    # last_games = dict(list(last_games.items())[:games])
    return last_games


def get_games_against_opponent(team_dict, team, opponent, date, start_date):
    selected_team = pd.DataFrame()
    # opposing_team = []
    # start_date = datetime.date(2022, 10, 17)
    delta = datetime.timedelta(days=-1)
    date += delta
    count = 0

    while (date > start_date):
        day = str(date.day)
        month = str(date.month)
        year = str(date.year)
        if (date.day < 10):
            day = "0" + day
        if (date.month < 10):
            month = "0" + month
        loop_date = '' + month + '-' + day + '-' + year
        for i, key in team_dict.items():
            if (i == loop_date):
                for item in team_dict[loop_date].keys():
                    if (item == team):
                        for x in team_dict[loop_date][item]:
                            if (x == opponent):
                                selected_team = pd.concat([selected_team, team_dict[loop_date][item][x]])
                                # opposing_team.append(team_dict[date][x][item])
                                count += 1
                                if (count >= 2):
                                    return selected_team

        date += delta
    if (count == 1):
        selected_team = pd.concat([selected_team, selected_team])
    if (count == 0):
        return pd.DataFrame()
    return selected_team


def process_seasondf(df):
    df.rename(columns={'Player': 'Starters', 'OFFRTG': 'ORtg', 'DEFRTG': 'DRtg'}, inplace=True)

    df_rosters = {}
    team_list = list(df.Tm.unique())
    tms = df.groupby('Tm')

    for i in team_list:
        df_rosters[i] = tms.get_group(i)

    df_sort = team21_22['10-30-2021']['CLE']['PHO']

    for idx in df_rosters:
        df_rosters[idx] = df_rosters[idx].sort_values(by='MP', ascending=False)
        df_rosters[idx] = df_rosters[idx].reindex(df_sort.columns, axis=1)
        df_rosters[idx] = df_rosters[idx].head(12)

    return df_rosters


def shape_data(team_dict, df_rosters, end_date):
    arrays = []
    wins = []

    for date, key in team_dict.items():
        for item in key.keys():
            for x in team_dict[date][item]:
                form = '%m-%d-%Y'
                new_date = datetime.datetime.strptime(date, form).date()
                if (get_games_against_opponent(team_dict, item, x, new_date, end_date).empty or find_recent_games(
                        team_dict, item, new_date, end_date).empty):
                    print('not enough games')
                    continue
                against_df = get_games_against_opponent(team_dict, item, x, new_date, end_date)
                recent = find_recent_games(team_dict, item, new_date, end_date)
                pad_df = pd.DataFrame(0, index=range(3), columns=recent.columns)
                df = pd.concat([against_df, pad_df, recent, pad_df, df_rosters[item]])
                arrays.append(df)
                if (team_dict[date][item][x].loc[12]['PTS'] > team_dict[date][x][item].loc[12]['PTS']):
                    wins.append(1)
                else:
                    wins.append(0)
                print(item + ' vs ' + x + ' array complete')
        print('' + str(date) + ' day done')

    return arrays, wins


df_roster20_21 = process_seasondf(df20_21)
df_roster21_22 = process_seasondf(df21_22)
df_roster18_19 = process_seasondf(df18_19)
df_roster17_18 = process_seasondf(df17_18)
df_roster16_17 = process_seasondf(df16_17)
df_roster15_16 = process_seasondf(df15_16)
test_df = process_seasondf(test_df)


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
            if (new_date >= dateleft and new_date < dateright):
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


team_dict20_21 = power_rank(ranking20_21, team_dict20_21, '05-16-2021')
team_dict21_22 = power_rank(ranking21_22, team_dict21_22, '04-10-2022')
team_dict18_19 = power_rank(ranking18_19, team_dict18_19, '04-10-2019')
team_dict17_18 = power_rank(ranking17_18, team_dict17_18, '04-11-2018')
test = power_rank(test_ranking, test, '04-09-2023')

team_dict16_17 = standings_rank(team_dict16_17, ranking16_17)
team_dict15_16 = standings_rank(team_dict15_16, ranking15_16)

arrays18_19, wins18_19 = shape_data(team_dict18_19, df_roster18_19, datetime.date(2018, 10, 15))
arrays17_18, wins17_18 = shape_data(team_dict17_18, df_roster17_18, datetime.date(2017, 10, 16))
testarrays, testwins = shape_data(test, test_df, datetime.date(2022, 10, 17))
arrays21_22, wins21_22 = shape_data(team_dict21_22, df_roster21_22, datetime.date(2021, 10, 18))
arrays20_21, wins20_21 = shape_data(team_dict20_21, df_roster20_21, datetime.date(2020, 10, 21))
arrays16_17, wins16_17 = shape_data(team_dict16_17, df_roster16_17, datetime.date(2016, 10, 24))
arrays15_16, wins15_16 = shape_data(team_dict15_16, df_roster15_16, datetime.date(2015, 10, 26))

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))


def final_shape(arrays):
    x_list = []
    for array in arrays:
        array = array[array['Starters'] != 'Team Totals']
        array = array.drop(['Starters'], axis=1)
        array['MP'] = array.loc[:]['MP'].apply(convert_MP)
        array = array.fillna(0)
        array = array.astype(float)
        array = min_max_scaler.fit_transform(array)
        x_list.append(array)
    return x_list


x_test = final_shape(testarrays)
x20_21 = final_shape(arrays20_21)
x21_22 = final_shape(arrays21_22)
x18_19 = final_shape(arrays18_19)
x17_18 = final_shape(arrays17_18)
x16_17 = final_shape(arrays16_17)
x15_16 = final_shape(arrays15_16)

x_list = x21_22 + x20_21 + x18_19 + x17_18 + x16_17 + x15_16

flattened_arrays = []
for game_data in x_list:
    flattened_game = []
    for team_data in game_data:
        flattened_game.extend(team_data.flatten())  # Flattening each team's DataFrame
    flattened_arrays.append(flattened_game)

len(x15_16)

arrays_to_process = [x21_22, x18_19, x17_18, x16_17, x15_16]

# Determine the minimum number of samples across all arrays
min_samples = min(len(arr) for arr in arrays_to_process)

# Drop data points to make all arrays the same shape
for idx, array in enumerate(arrays_to_process):
    arrays_to_process[idx] = array[:min_samples]

# Concatenate the processed arrays into a single array
X_combined = np.concatenate(arrays_to_process, axis=0)

X_combined.shape

wins_to_process = [wins21_22, wins18_19, wins17_18, wins16_17, wins15_16]

# Determine the minimum number of samples across all 'wins' arrays
min_samples = min(len(wins) for wins in wins_to_process)

# Drop data points to make all 'wins' arrays the same shape
for idx, wins_array in enumerate(wins_to_process):
    wins_to_process[idx] = wins_array[:min_samples]

# Concatenate the processed 'wins' arrays into a single array
y_combined = np.concatenate(wins_to_process, axis=0)

y_combined.shape

X = np.array(flattened_arrays)

X.shape

# X = X.reshape(X.shape[0], -1)
for i in range(len(X)):
    X[i] = np.reshape(np.transpose(X[i]), 162 * 37)

X = X.reshape(X.shape[0], -1)

X.shape

x_test = np.array(x_test)

x_test.shape

x_test = x_test.reshape(x_test.shape[0], -1)

x_test.shape

wins_test = np.array(testwins)

wins = wins21_22 + wins20_21 + wins18_19 + wins17_18 + wins16_17 + wins15_16

y = np.array(wins)

y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

y_train.shape

eval_set = [(X_train, y_train), (X_test, y_test)]


def objective(trial, data=X_train, target=y_train):
    param = {
        'tree_method': 'gpu_hist',
        # this parameter means using the GPU when training our model to speedup the training process
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 1.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': trial.suggest_categorical('n_estimators', [2000, 3000, 3500, 4000, 5000, 6000]),
        'max_depth': trial.suggest_categorical('max_depth', [1, 3, 5, 7, 9, 11, 13, 15, 17]),
        # 'random_state': trial.suggest_categorical('random_state', [2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
    }
    model = xgb.XGBClassifier(**param)

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(x_test)

    accuracy = accuracy_score(wins_test, preds)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
trial = study.best_trial
print("  Value: ", trial.value)

# best_params = study.best_trial.params
params = {'lambda': 0.34649586581992475, 'alpha': 0.418230742143025, 'colsample_bytree': 0.8, 'subsample': 0.8,
          'learning_rate': 0.016, 'n_estimators': 2000, 'max_depth': 17, 'min_child_weight': 2,
          'gamma': 6.422333220506358e-08}
print(params)

model = xgb.XGBClassifier(**params, tree_method='gpu_hist', feature_selector='shuffle')

model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

predictions = model.predict(x_test)

accuracy = accuracy_score(wins_test, predictions)
print("Accuracy:", accuracy)

precision = precision_score(wins_test, predictions, average='weighted')
recall = recall_score(wins_test, predictions, average='weighted')
f1 = f1_score(wins_test, predictions, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

feature_importance = model.feature_importances_
print(feature_importance)

score_dict = model.get_booster().get_score(importance_type='gain')

scores_list = list(score_dict.keys())
scores_list

col_list = []
for col in arrays21_22[0].columns:
    if col != 'Starters':
        col_list.append(col)

col_list

count = 0
sco = {}

for i in range(37):
    sco[i] = []

for key, val in score_dict.items():
    if (count >= 37):
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

results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

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

# opt = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size=128)

# train_loss, train_accuracy = model.evaluate(X_train, y_train)

# print(f"Training Accuracy: {train_accuracy:.4f}")






