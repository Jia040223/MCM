import pandas as pd
import numpy as np
import csv

__path__ = './Data/'

class data_loader():
    def __init__(self,__path__,fliename):
        self.__path__ = __path__
        self.df = pd.read_csv(__path__ + fliename)
    
    def find_match_player(self):
        df=self.df
        match_set = set()
        for index ,row in df.iterrows():
            match_set.add((row['player1'],row['player2']))
        return match_set
        

def cal_init_prob(match_set):
    df_elo = pd.read_csv(__path__ + 'players_elo_complete.csv')
    match_prob={}
    for match in match_set:
        player1 = match[0]
        player2 = match[1]
        elo1 = df_elo[df_elo['name'] == player1]['elo'].astype(int).iloc[0]
        elo2 = df_elo[df_elo['name'] == player2]['elo'].astype(int).iloc[0]
        p1=elo1/(elo1+elo2)
        p2=elo2/(elo1+elo2)
        match_prob[match] = (p1,p2)
    return match_prob

def Get_Match_Init_Probs():
    dat_loader = data_loader(__path__,'Wimbledon_featured_matches.csv')
    match_set = dat_loader.find_match_player()
    match_prob = cal_init_prob(match_set)

    return match_prob

    