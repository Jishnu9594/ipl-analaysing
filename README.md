# ipl-analaysing
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import warnings
from statistics import median

import matplotlib.pyplot as plt

%matplotlib inline
warnings.filterwarnings('ignore')
df = pd.read_csv("ipl_2019_batting_partnerships.csv")
df.head()
# prepare dataframe for Delhi Capitals
df_dc = df[df['team']=="Delhi Capitals"]

df_dc['partners'] = [sorted([i,j]) for i,j in zip(df_dc['player_1'], df_dc['player_2'])]
df_dc['partnership'] = ["".join(i) for i in df_dc['partners']]

df_dc.head()
# empty list to store players name
p1 = []
p2 = []

# empty lists to store median of runs scored
r1 = []
r2 = []

for p in df_dc['partnership'].unique():
    
    temp = df_dc[df_dc['partnership'] == p]
    p1.append(temp.iloc[0]['player_1'])
    p2.append(temp.iloc[0]['player_2'])
    
    a = []
    b = []
    
    # extract individual scores for both the players
    for index, row in temp.iterrows():
        # scores of player 1
        a.append(row['score_1'])
        
        # scores of player 2
        b.append(row['score_2'])

    # append median of scores    
    r1.append(median(a))
    r2.append(median(b))
    # find the leading batsman
team_df['lead'] = np.where(team_df['r1'] >= team_df['r2'], team_df['p1'], team_df['p2'])
team_df['follower'] = np.where(team_df['lead'] == team_df['p1'], team_df['p2'], team_df['p1'])
team_df['larger_score'] = np.where(team_df['r1'] >= team_df['r2'], team_df['r1'], team_df['r2'])
team_df['total_score'] = team_df['r1'] + team_df['r2']

# performance ratio
team_df['performance'] = team_df['larger_score']/(team_df['total_score']+0.01)
# construct graph
G = nx.from_pandas_edgelist(team_df, "follower", "lead", ['performance'], create_using=nx.MultiDiGraph())

# get edge weights
_, wt = zip(*nx.get_edge_attributes(G, 'performance').items())
